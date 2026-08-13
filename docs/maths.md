# Fixed-Point Arithmetic of the FPGA Matmul Cores

This document explains the exact integer/fixed-point arithmetic performed by the
Q8_0 and Q5_0 compute cores (`matmul_q8_core.v`, `matmul_q5_0_core.v`). It is the
authoritative reference for the bit-exact golden models in `sim/golden_model.hpp`.

Every accumulator, scale, and intermediate value is an **integer**; the "point"
(scale factor) only exists in how we interpret and convert those integers. The
cores never see floats.

---

## 1. Fixed-Point Notation

A fixed-point number stores a real value as an integer with an implicit binary
point:

```
stored_integer = real_value × 2^(fractional_bits)
```

| Notation | Total bits (sign + int + frac) | Stored range | Real range | `1.0` stored as |
|----------|--------------------------------|--------------|------------|------------------|
| **S8**   | 8  (1 + 7 + 0)                | −128 … 127   | −128 … 127 | 1 |
| **S5**   | 5  (1 + 4 + 0)                | −16 … 15     | −16 … 15   | 1 |
| **S16**  | 16 (1 + 15 + 0)               | −32768 … 32767 | −32768 … 32767 | 1 |
| **S24.8**| 32 (1 + 23 + 8)               | ±8388608     | ±32768.0   | 256 |
| **S24.16**| 32 (1 + 15 + 16)              | ±32768       | ±32768.0   | 65536 |
| **UQ8.8**| 16 (8 + 8, unsigned)          | 0 … 65535    | 0 … 255.996| 256 |

Key facts:

- **S16** is a plain signed integer. Activations and clamped weights are S16.
- **S24.8** and **UQ8.8** both store `1.0` as the integer **256** (8 fractional
  bits). `>> 8` divides an integer by 256 — i.e. it drops the fractional bits.
- `x >>> 8` in Verilog is an **arithmetic** shift (sign-extending, rounds toward
  −∞). For non-negative operands it is identical to logical `>> 8`.
- A value "clamped to S16" means: if it exceeds `32767` → `32767`; if below
  `−32768` → `−32768`; otherwise keep the low 16 bits.

---

## 2. Q8_0 Core

Q8_0 dequantizes each 8-bit weight by a per-block scale, then does a dot product
with the activation.

### Data widths

| Operand | Width | Meaning |
|---------|-------|---------|
| `q8`    | S8    | raw quantized weight (−128 … 127) |
| `sc`    | UQ8.8 | dequantization scale, `sc = combined × 256` |
| `act`   | S16   | quantized activation |
| `deq`   | S16   | dequantized weight (reconstructed, integer units) |
| `acc`   | S48   | accumulator |

### Operations

```
deq = (q8 × sc) >>> 8          // arithmetic shift, then truncate to S16
acc = Σ (deq × act)            // 48-bit signed
```

- `q8 × sc` is an 8-bit × 16-bit signed product (max magnitude `127 × 65535
  = 8,322,945`, well within 32 bits).
- `>>> 8` = divide by 256 = `q8 × (sc/256) = q8 × real_scale`.
- **No saturation is needed** after the shift: the result always fits S16
  (max ≈ 32511, min ≈ −32768). The function return type `signed [15:0]` performs
  a plain truncation (this is documented in AGENTS.md — the old explicit
  saturate was removed because it can never fire).
- `deq × act` is a 16×16 signed product; the sum over ≤64 columns fits the 48-bit
  accumulator (no overflow for real model data).

Because the `>>> 8` happens **before** the MAC, the `256×` factor is removed up
front. The accumulator is therefore in **plain integer units**
(real_weight × int_activation), and the host conversion needs **no `/256`**:

```
y = acc × x_scale × row_scale
```

### Worked example

Take one term: `q8 = 50`, `sc = 512` (= 2.0 in UQ8.8), `act = 100`.

```
q8 × sc    = 50 × 512 = 25600
>>> 8      = 100                        → 50 × 2.0 = 100  ✓
deq × act  = 100 × 100 = 10000          (this one term's contribution)
```

---

## 3. Q5_0 Core

Q5_0 dequantizes 5-bit weights by a per-block f16 scale `d`, optionally scaled by
a per-row normalization `norm`, then does the dot product.

### Data widths

| Operand | Width | Meaning |
|---------|-------|---------|
| `d`     | f16   | GGUF block scale (header, stored as 16-bit half) |
| `f16_decode(d)` | S24.8 | `1.0` → 256 |
| `norm`  | UQ8.8 | per-row normalization, `1.0` → 256 |
| `d_pre` | S16   | reconstructed scale **after clamping** |
| `q5`    | S5    | signed 5-bit weight (−16 … 15) |
| `dq`    | 21-bit signed | dequantized weight (S24.16 scale) |
| `act`   | S16   | quantized activation |
| `acc`   | S48   | accumulator |

### Operations

```
d_pre = (f16_decode(d) × norm) >>> 8      // then clamp to S16
dq    = d_pre × q5                        // 21-bit signed (16 × 5)
acc   = Σ (dq × act)                      // 48-bit signed
```

- `f16_decode(d)` returns S24.16: the half-precision value scaled by 65536.
  See §4 for the exact bit-level algorithm.
- `d_pre = f16_decode(d) × norm / 256`. With `norm = 256` (=1.0),
  `d_pre = f16_decode(d)` — the S24.16 value, then clamped to S16 (±32767,
  i.e. block scales `d ≤ 0.5`).
- `dq = d_pre × q5` is in **S24.16**: `dq / 65536 = d × q5 × (norm/256)` is the
  real dequantized weight.
- `dq × act` is a 21×16 signed product (37 bits); summed over 896 columns into
  the 48-bit accumulator.

Because `d_pre` stays in S24.16 (the `×65536` survives into `dq`), the accumulator
is in **S24.16** (real value × 65536), and the host conversion needs a **`/65536`**:

```
y = acc × x_scale / 65536
```

### Worked example

`d = 0.25` (f16), `norm = 256` (=1.0), `q5 = 5`, `act = 100`.

```
f16_decode(0.25) = 16384                  (0.25 × 65536)
d_pre = (16384 × 256) >>> 8 = 16384       (0.25, no clamp needed)
dq    = 16384 × 5 = 81920                 (81920/65536 = 1.25 = 0.25 × 5  ✓)
dq × act = 81920 × 100 = 8192000          (8192000/65536 = 125 = 0.25 × 5 × 100  ✓)
```

---

## 4. `f16_decode` — exact bit-level algorithm

Reproduced verbatim from `matmul_q5_0_core.v` (function `f16_decode`) and the
golden model. Input is a 16-bit half; output is **S24.16** (16 fractional bits).

```c
// input: uint16_t f16;  output: int32_t (S24.16, 1.0 -> 65536)
int32_t exp  = (f16 >> 10) & 0x1F;      // 5-bit exponent
int32_t mant = f16 & 0x3FF;             // 10-bit mantissa

if (exp == 0 || exp == 31) return 0;                      // subnormal/inf/NaN -> 0
if (exp >= 9) return (1024 + mant) << (exp - 9);          // exact left shift
return ((1024 + mant) + (1 << (8 - exp))) >> (9 - exp);   // round-half-up, then shift
```

Notes:

- **Sign-agnostic**: bit 15 (the half's sign) is ignored — Q5_0 block scales are
  defined non-negative in GGUF.
- The output is the half's magnitude scaled to S24.16: `value × 65536`.
- The `else` branch adds `2^(8−exp)` before shifting right `9−exp` bits, i.e.
  round-half-up.
- `f16_decode(1.0)` (0x3C00) = `(1024 + 0) << (15 − 9)` = `1024 << 6` = 65536.
- **16 fractional bits** (was S24.8 / 8 bits, which rounded block scales
  `d < 0.004` to zero). With 16 bits, `d = 0.001` → ~66, preserving small Q5_0
  block scales.

---

## 5. Structural difference: where the scale factor lives

| | factor removed where? | Accumulator units | Host conversion |
|--|---------------------|-------------------|-----------------|
| **Q8** | `deq` does `>>> 8` **before** the MAC | plain integer | `acc × x_scale × row_scale` |
| **Q5** | `d_pre` does `>>> 8`, but `dq` re-multiplies by `q5` keeping the ×65536 | **S24.16** (×65536) | `acc × x_scale / 65536` |

This is the single most common source of off-by-scale-factor confusion. When
verifying a core against a reference, first confirm the accumulator scaling matches
this table before hunting for address/layout bugs.

---

## 6. Accuracy implications (the `d_pre` fractional-precision fix)

`d_pre` is **S16** (`reg signed [15:0] d_pre`, `matmul_q5_0_core.v:87`), and
`d_pre = f16_decode(d) × norm >> 8 = d × 65536 × (norm/256)`, clamped to ±32767.

The original design used **S24.8** `f16_decode` (8 fractional bits): a block scale
`d ≈ 0.001` gave `round(0.001 × 256) = 0` → `d_pre = 0` → the whole 32-weight
block dequantized to zero. Q5_0 block scales of `1e-3 … 1e-1` are typical, so many
blocks lost all precision (the source of the ~5-22 `maxdiff` vs the CPU).

**Fix (2026-08-13):** `f16_decode` now outputs **S24.16** (16 fractional bits), so
`d = 0.001` → ~66 instead of 0. This is a **zero-resource** change — only the shift
amounts in `f16_decode` move; `d_pre` stays S16, so the `dq` LUT multiply (16×5)
and the DSP MAC (21×16) keep their exact widths. The output scaling changes from
`/256` to `/65536`.

Remaining trade-off: with `norm = 1.0`, block scales `d > 0.5` saturate `d_pre` at
S16 (±32767). This is acceptable for Q5_0 (scales are small); if larger scales
appear, the per-row `norm` (UQ8.8) can be reduced to bring them into range.

The Q8 core has no equivalent bottleneck: its `sc` is UQ8.8 (wider, 0…65535) and
the `>>> 8` is applied per-weight before the MAC, so `deq` is already in integer
units and always fits S16.
