# HLS Q8 Kernel Line-by-Line Explanation

**File**: `hls/matmul_q8.cpp`
**Target**: Zynq 7010 (80 DSP, 17,600 LUT, 135 KB BRAM)
**Kernel**: Q8_0 direct-path vector-matrix / matrix multiply with LUT-based scale multipliers

---

## Lines 1-4: Includes

```cpp
#include <hls_stream.h>       // HLS stream types (vestigial — not used)
#include <ap_int.h>            // Arbitrary-width integers: ap_int<N>, ap_uint<N>
#include <stdint.h>            // Standard C int types (uint64_t, etc.)
#include "matmul_q8.hpp"       // N_Q8=64, BLOCK_Q8=8, Q8_BLOCK_SIZE=32, etc.
```

`hls_stream.h` is for AXI-Stream interfaces (not used in this kernel — AXI-MM instead). `ap_int.h` is the core HLS type: `ap_int<16>` is a 16-bit signed integer synthesizable to hardware. `matmul_q8.hpp` defines the interface constants shared between HLS and the testbench.

---

## Lines 6-11: Tile and Q8 block constants

```cpp
constexpr int N = 64;                 // Tile size: 64×64
constexpr int BLOCK = 8;             // Systolic sub-block: 8×8
constexpr int NUM_BLOCKS = N / BLOCK; // 8 sub-blocks per dimension
constexpr int Q8_BLOCK_SIZE = 32;    // Q8_0: 32 elements per FP16 scale
constexpr int Q8_BLOCK_BYTES = 34;   // Q8_0: 2 (FP16 scale) + 32 (INT8 data) bytes
constexpr int ROWS_PER_SCALE = 1;    // Each row has its own scale (not shared)
```

**N=64**: The tile size for the systolic array. The matmul is blocked into 64×64 tiles. For a 1×896 × 896×N matmul (decode), each tile is 1×64 × 64×64 producing 1×64 partial results. For the full 64×64 matmul (prefill), all 4096 MACs run in parallel.

**BLOCK=8**: The systolic sub-block. 8×8 = 64 DSPs per cycle. The full 64×64 matmul takes 8 outer-product iterations.

**NUM_BLOCKS=8**: 64 / 8 = 8 sub-blocks along each axis.

**Q8_BLOCK_SIZE=32**: Standard Q8_0 format: 32 elements share one FP16 scale.

**Q8_BLOCK_BYTES=34**: Each Q8_0 block = 2 bytes (FP16 scale) + 32 bytes (INT8 values).

---

## Lines 13-16: AXI-Lite status values

```cpp
constexpr int STATUS_IDLE    = 0;
constexpr int STATUS_RUNNING = 1;
constexpr int STATUS_DONE    = 2;
constexpr int STATUS_ERROR   = 3;
```

These are written to the AXI-Lite status register so the ARM can poll or interrupt on completion.

---

## Lines 18-25: Type definitions

```cpp
typedef ap_int<16> in_t;               // INT16: activation + dequantized weight
typedef ap_int<32> prod_t;             // INT16 × INT16 = 32-bit product
typedef ap_int<64> acc_t;              // 64-deep dot product => 64-bit accumulator

typedef ap_uint<16> combined_scale_t;  // UQ8.8 unsigned fixed-point
constexpr int SCALE_FRAC_BITS = 8;     // Right-shift to remove fractional bits
```

**Why acc_t is 64-bit**: The dot product of 64 INT16 values:
- Max single product: 32767² = 1,073,709,056 (31 bits)
- Sum of 64: 64 × 1.07×10⁹ = 6.87×10¹⁰ (37 bits)
- Residual accumulation across tiles can sum multiple 64-tile results
- 64-bit accumulator guarantees no overflow

**combined_scale_t (UQ8.8)**: 8 integer bits, 8 fractional bits. Range [0, 255.996], precision 1/256 = 0.0039. The combined scale = `block_scale / row_scale`, precomputed as fixed-point on ARM.

---

## Lines 27-52: `f16_bits_to_uq8_8()` — FP16 bit pattern to UQ8.8 fixed-point

```cpp
inline combined_scale_t f16_bits_to_uq8_8(ap_uint<16> f16_bits) {
#pragma HLS INLINE
```

This function takes a raw FP16 bit pattern (the scale stored in Q8_0 format) and produces a UQ8.8 fixed-point approximation. It is `INLINE` — the entire function body is instantiated at each call site with no function call overhead.

### Bitfield extraction (lines 30-32):

```cpp
    ap_uint<1> sign = f16_bits[15];           // bit 15: sign (0=positive, 1=negative)
    ap_uint<5> exp = f16_bits.range(14, 10);  // bits 14-10: exponent (biased by 15)
    ap_uint<10> mant = f16_bits.range(9, 0);  // bits 9-0: mantissa (10 bits)
```

FP16 format:
```
15     14      10 9           0
[sign] [exponent] [mantissa]
  1b       5b        10b
```

Value = (-1)^sign × 2^(exp-15) × (1 + mant/1024)  for normalized numbers (exp > 0)
Value = (-1)^sign × 2^(-14) × (mant/1024)         for subnormals (exp = 0)

### Case 1: Positive normalized, value >= 1 (lines 36-42):

```cpp
    if (sign == 0) {                         // positive
        if (exp >= 15) {                     // exp >= 15 => value >= 1.0
            ap_uint<8> int_part = (1 << (exp - 15));  // integer part: 2^(exp-15)
            ap_uint<8> frac_part = (exp >= 23) ? 0 :
                (mant >> (10 - (exp - 15)));  // mantissa bits shifted into fractional
            result.range(15, 8) = int_part;   // high byte = integer
            result.range(7, 0) = frac_part;   // low byte = fractional
        }
```

Example: exp=15, mant=0 => value=1.0 => int_part=1<<0=1, frac=0 => UQ8.8=0x0100
Example: exp=16, mant=0x200 (0.5) => value=2.5 => int_part=1<<1=2, frac=mant>>(10-1)=0x200>>9=1 => UQ8.8=0x0201

### Case 2: Positive subnormal, value in (0, 1) (lines 43-48):

```cpp
        } else if (exp >= 7) {               // exp in [7,14]: value between 0 and 1
            result.range(15, 8) = 0;          // integer part = 0
            ap_uint<8> frac = (1 << (exp - 7)) |  // leading 1 (implicit in normalized)
                              (mant >> (10 - (exp - 7)));
            result.range(7, 0) = frac;
        }
```

For subnormals (exp=0), there's no implicit leading 1. The formula is `mant/1024 × 2^(-14)`. This is very small (< 2^(-14) ≈ 6×10⁻⁵), and our combined scale is rarely this small, so `exp >= 7` is a reasonable lower bound.

### Line 51: Return

```cpp
    return result;
}
```

The entire conversion uses only bitfield extraction and shifts — no multipliers. Pure LUT logic.

---

## Lines 54-63: `q8_dequant_lut()` — Scale multiply (core operation)

```cpp
inline in_t q8_dequant_lut(ap_int<8> q8_val, combined_scale_t scale) {
#pragma HLS INLINE
```

Called once per weight element. Converts Q8_0 INT8 → INT16 using the combined UQ8.8 scale.

### Line 58: INT8 × UQ8.8 multiply

```cpp
    ap_int<24> product = (ap_int<24>)q8_val * (ap_int<24>)scale;
```

8-bit signed × 16-bit unsigned → 24-bit signed. The multiplication is forced to LUTs by `config_bind -mul_style luts` in `script_q8.tcl`. A 24×24 multiplier consumes ~24 LUT6 + carry chain = ~48 LUTs per multiplier, or 64×48 = ~3,072 LUTs total for all 64 parallel dequant units.

### Line 59: UQ8.8 → INT16 by right-shift

```cpp
    ap_int<16> val = (ap_int<16>)(product >> SCALE_FRAC_BITS);  // >> 8
```

The combined scale is in UQ8.8 format: `combined_fixed = combined_float × 256`. So:
```
result = (q8_val × combined_fixed) >> 8
       = q8_val × combined_float × 256 / 256
       = q8_val × combined_float
       = q8_val × block_scale / row_scale
```

The right-shift by 8 removes the fixed-point scaling, producing the dequantized INT16 value.

### Lines 60-62: Clamp to INT16 range

```cpp
    if (val > 32767) val = 32767;
    if (val < -32768) val = -32768;
    return val;
```

Product range: q8_val ∈ [-128, 127], combined ∈ [0, 65535]
- product ∈ [-128×65535, 127×65535] = [-8,388,480, 8,321,745]
- After >> 8: [-32767, 32506]
- Lower bound hits -32768 only if q8_val=-128 AND combined > 32768, which requires combined_float > 128 — possible for small row_scale values
- Upper bound hits 32767 only if q8_val=127 AND combined > 32768 — same condition

The clamp is a safety net; in practice combined rarely exceeds 65535 (UQ8.8 max).

---

## Lines 65-115: `systolic_array_8x8()` — 8×8 INT16 systolic array

This is the compute engine: 8 rows × 8 columns of DSP48E1 blocks doing `C += A × B`.

### Lines 68-78: Array partitioning

```cpp
void systolic_array_8x8(in_t A[BLOCK][BLOCK], in_t B[BLOCK][BLOCK],
                         acc_t C[BLOCK][BLOCK]) {
    #pragma HLS ARRAY_PARTITION variable=A complete dim=1  // rows → 8 registers each
    #pragma HLS ARRAY_PARTITION variable=B complete dim=2  // cols → 8 registers each
    #pragma HLS ARRAY_PARTITION variable=C complete         // all 64 = individual FF

    in_t a_reg[BLOCK][BLOCK];    // local register array for A
    in_t b_reg[BLOCK][BLOCK];    // local register array for B
    #pragma HLS ARRAY_PARTITION variable=a_reg complete
    #pragma HLS ARRAY_PARTITION variable=b_reg complete

    acc_t c_reg[BLOCK][BLOCK];   // local accumulator array
    #pragma HLS ARRAY_PARTITION variable=c_reg complete
```

**`complete` partitioning** = every array element is a separate flip-flop. For `c_reg[8][8]`, that's 64 × 64-bit FFs = 4,096 FFs. This allows all 64 elements to be read/written simultaneously in one cycle.

### Lines 80-85: Zero accumulators

```cpp
    for (int i = 0; i < BLOCK; i++)
        for (int j = 0; j < BLOCK; j++) {
            #pragma HLS UNROLL              // fully unrolled
            c_reg[i][j] = 0;
        }
}
```

Fully unrolled (64 parallel writes), zero all accumulators in one cycle.

### Lines 87-96: Load A and B into local registers

```cpp
    for (int k = 0; k < BLOCK; k++) {
        for (int i = 0; i < BLOCK; i++) {
            #pragma HLS UNROLL
            a_reg[i][k] = A[i][k];       // load column k of A
        }
        for (int j = 0; j < BLOCK; j++) {
            #pragma HLS UNROLL
            b_reg[k][j] = B[k][j];       // load row k of B
        }
    }
```

This loads the data from input arguments to local registers. Each iteration of `k` loads one "column slice" of A (8 elements) and one "row slice" of B (8 elements) — 16 total loads, all parallel. 8 iterations × 1 cycle = 8 cycles to load all 128 elements.

### Lines 98-107: The systolic compute loop

```cpp
    for (int t = 0; t < BLOCK; t++) {
        for (int i = 0; i < BLOCK; i++) {
            for (int j = 0; j < BLOCK; j++) {
                #pragma HLS UNROLL          // 64 parallel MACs, fully unrolled
                prod_t a_val = (prod_t)a_reg[i][t];   // row i, column t
                prod_t b_val = (prod_t)b_reg[t][j];   // row t, column j
                c_reg[i][j] += (acc_t)(a_val * b_val); // MAC
            }
        }
    }
```

This is the **outer-product accumulation**. At each `t` (0..7):
- All 64 MACs execute in parallel (i=0..7, j=0..7)
- Each MAC: `c_reg[i][j] += A[i][t] × B[t][j]`
- 8 cycles total for the full 8×8×8 = 512 MACs

The DSP48E1 on Zynq 7010 implements each MAC as: `P = A × B + CIN` where P is 48-bit. 64 DSPs × 8 cycles = 512 MACs/tile.

### Lines 109-114: Write results

```cpp
    for (int i = 0; i < BLOCK; i++)
        for (int j = 0; j < BLOCK; j++) {
            #pragma HLS UNROLL
            C[i][j] = c_reg[i][j];
        }
}
```

64 parallel writes, 1 cycle. Total: 1 (zero) + 8 (load) + 8 (compute) + 1 (write) = 18 cycles for a full 8×8 MAC.

---

## Lines 117-188: `matmul_64x64()` — Blocked 64×64 matmul

This decomposes a 64×64 matmul into 8×8 sub-blocks and sequences them through `systolic_array_8x8`.

### Line 118: DATAFLOW pragma

```cpp
void matmul_64x64(in_t A[N][N], in_t B[N][N], acc_t C[N][N]) {
    #pragma HLS DATAFLOW
```

`DATAFLOW` enables function-level pipelining. HLS automatically creates Ping-Pong buffers between the three stages (load A, load B, compute + write), so they can overlap. While sub-block (p,q) is computing, sub-block (p+1,q) is loading.

### Lines 120-123: Sub-block buffers

```cpp
    in_t A_buf[NUM_BLOCKS][BLOCK][N];     // 8 blocks × 8 rows × 64 cols
    in_t B_buf[NUM_BLOCKS][BLOCK][N];     // 8 blocks × 8 rows × 64 cols
    #pragma HLS ARRAY_PARTITION variable=A_buf cyclic factor=2 dim=2  // 2-bank on rows
    #pragma HLS ARRAY_PARTITION variable=B_buf cyclic factor=2 dim=2
```

`A_buf[p][i][k]` stores the full 64-element row `k` for the i-th 8-row group in sub-block p. The cyclic partitioning on dim=2 (rows within a block) with factor 2 means rows 0,2,4,6 are in bank 0 and rows 1,3,5,7 in bank 1 — 2-parallel read when the inner loop unrolls.

### Lines 125-132: Load A

```cpp
    for (int p = 0; p < NUM_BLOCKS; p++)
        for (int i = 0; i < BLOCK; i++)
            for (int k = 0; k < N; k++) {
                #pragma HLS PIPELINE II=1
                A_buf[p][i][k] = A[p * BLOCK + i][k];
            }
```

Copies the full 64×64 A matrix into the local buffer. `p * BLOCK + i` selects the row from the input, `k` iterates over all 64 columns. Pipelined at `II=1` (one element/cycle): 8×8×64 = 512 cycles.

### Lines 134-141: Load B

```cpp
    for (int q = 0; q < NUM_BLOCKS; q++)
        for (int k = 0; k < BLOCK; k++)
            for (int j = 0; j < N; j++) {
                #pragma HLS PIPELINE II=1
                B_buf[q][k][j] = B[q * BLOCK + k][j];
            }
```

Same pattern for B: 512 cycles.

### Lines 143-145: Partial product storage

```cpp
    acc_t P[NUM_BLOCKS][NUM_BLOCKS][BLOCK][BLOCK];
    #pragma HLS ARRAY_PARTITION variable=P complete dim=3  // 8×8 per sub-block accessible
    #pragma HLS ARRAY_PARTITION variable=P complete dim=4
```

`P[p][q][i][j]` = partial product for sub-block pair (p,q), row i, column j. Dimensions 3 and 4 are partitioned completely so each 8×8 × 64-bit can be read/written in one cycle.

### Lines 147-176: Sub-block computation

```cpp
    for (int p = 0; p < NUM_BLOCKS; p++) {
        for (int q = 0; q < NUM_BLOCKS; q++) {
            in_t A_block[BLOCK][BLOCK];     // 8×8 sub-block of A
            in_t B_block[BLOCK][BLOCK];     // 8×8 sub-block of B
            acc_t C_block[BLOCK][BLOCK];

            // Extract A_block from A_buf
            for (int i = 0; i < BLOCK; i++)
                for (int k = 0; k < BLOCK; k++)
                    A_block[i][k] = A_buf[p][i][q * BLOCK + k];

            // Extract B_block from B_buf
            for (int k = 0; k < BLOCK; k++)
                for (int j = 0; j < BLOCK; j++)
                    B_block[k][j] = B_buf[q][k][j];

            systolic_array_8x8(A_block, B_block, C_block);

            // Store partial result
            for (int i = 0; i < BLOCK; i++)
                for (int j = 0; j < BLOCK; j++)
                    P[p][q][i][j] = C_block[i][j];
        }
    }
```

For each sub-block pair (p,q):
- `A_block[i][k]` = from buffer block p, row i, column `q*8 + k`
- `B_block[k][j]` = from buffer block q, row k, column j
- Run `systolic_array_8x8(A_block, B_block, C_block)` — 18 cycles
- Store 8×8 partial product into P

64 sub-block pairs × 18 cycles = 1152 cycles. With DATAFLOW pipelining, the next sub-block's load overlaps with current compute, reducing effective latency.

### Lines 178-187: Stitch partials into full result

```cpp
    for (int p = 0; p < NUM_BLOCKS; p++)
        for (int i = 0; i < BLOCK; i++)
            for (int q = 0; q < NUM_BLOCKS; q++)
                for (int j = 0; j < BLOCK; j++) {
                    #pragma HLS PIPELINE II=1
                    C[p * BLOCK + i][q * BLOCK + j] = P[p][q][i][j];
                }
}
```

Write the 64 × 64 = 4096 results from the 8×8×8×8 partial product array into the flat output matrix C. Pipelined at II=1: 4096 cycles, but this is a non-recurring setup cost (only matters for prefill, not decode).

---

## Lines 190-233: `vecmul_1x64_q8()` — Q8 vector-matrix multiply (DECODE path)

This is the primary path for single-token generation (decode). It does Q8 dequant + INT16 MAC in a single pass, one column at a time.

### Lines 193-199: Interface and partitioning

```cpp
void vecmul_1x64_q8(ap_int<8> B_q8[N][N],                    // Q8 raw bytes (flat)
                    combined_scale_t combined_scales[N][2],   // UQ8.8 scales
                    in_t A[N],                                // INT16 activation
                    acc_t result[N]) {                        // INT64 results
    #pragma HLS ARRAY_PARTITION variable=A complete             // 64 registers
    #pragma HLS ARRAY_PARTITION variable=result cyclic factor=8 // 8 banks
    #pragma HLS ARRAY_PARTITION variable=B_q8 cyclic factor=8 dim=2  // 8 banks on cols
    #pragma HLS ARRAY_PARTITION variable=combined_scales complete dim=1
```

**`B_q8 cyclic factor=8 dim=2`**: The 64 columns of B_q8 are split into 8 banks modulo 8. Column 0 → bank 0, column 1 → bank 1, ..., column 8 → bank 0. When the inner loop unrolls by 8, each bank serves one column independently.

**`combined_scales complete dim=1`**: All 64 rows of combined_scales[i][0..1] are individual registers. Each row has 2 scales (one per Q8 block).

### Lines 201-207: Zero accumulators

```cpp
    acc_t row_acc[N];
    #pragma HLS ARRAY_PARTITION variable=row_acc cyclic factor=8

    for (int i = 0; i < N; i++) {
        #pragma HLS UNROLL factor=8       // 8 per cycle, 8 cycles total
        row_acc[i] = 0;
    }
```

### Lines 209-227: Main compute loop

```cpp
    for (int k = 0; k < N; k++) {         // column loop (input dimension)
        #pragma HLS PIPELINE II=1          // one column per cycle
        in_t a_val = A[k];                 // activation element k
```

**Outer loop**: iterates over the 64 input columns. Pipelined at `II=1`: one column processed per cycle.

Each iteration broadcasts activation `a_val` to all 64 rows.

```cpp
        for (int i = 0; i < N; i++) {     // row loop (output dimension)
            #pragma HLS UNROLL factor=8    // 8 rows per inner cycle
```

The inner loop is unrolled by 8. So across 8 clock cycles of the inner loop, all 64 rows are processed (8 rows × 8 cycles = 64 rows total per column).

Inside the unrolled body:

```cpp
            int block_idx = i / Q8_BLOCK_SIZE;          // 0 for i=0..31, 1 for i=32..63
            int offset_in_block = i % Q8_BLOCK_SIZE;    // 0..31

            ap_int<8> q8_val = B_q8[i][Q8_BLOCK_BYTES * block_idx + 2 + offset_in_block];
```

**BUG**: This indexing assumes B_q8 contains full Q8_0 blocks (with FP16 scale bytes), but the top-level loads B_mat as flat `A[i*N + j]` — 4096 raw INT8 bytes without scale bytes. The index `34*0 + 2 + 0..31` = `2..33` skips columns 0-1 and reads columns 2-33 instead of 0-31. For `block_idx=1`, index `34*1 + 2 + 0..31` = `36..67` reads out of bounds (max column index is 63).

**Correct index should be**: `B_q8[i][k]` — the raw INT8 value at row i, column k, since the combined scale already accounts for the block normalization.

```cpp
            combined_scale_t cs = combined_scales[i][block_idx];  // UQ8.8 for this row+block

            in_t b_val = q8_dequant_lut(q8_val, cs);     // (q8 × combined) >> 8 → INT16

            prod_t prod = (prod_t)a_val * (prod_t)b_val; // INT16 × INT16 → 32-bit
            row_acc[i] += (acc_t)prod;                    // accumulate in 64-bit
```

Per cycle (8 rows): 8 dequants + 8 MACs = 16 ops. Per column (64 rows): 64 dequants + 64 MACs = 128 ops. Per tile (64 cols): 4096 dequants + 4096 MACs = 8192 ops.

### Lines 229-232: Write results

```cpp
    for (int i = 0; i < N; i++) {
        #pragma HLS UNROLL factor=8
        result[i] = row_acc[i];
    }
}
```

---

## Lines 235-343: `matmul_q8()` — Top-level kernel

This is the entry point called by the ARM via AXI. It reads data from DDR, dispatches to vecmul or matmul, writes results back.

### Lines 237-253: Function signature and AXI interface declarations

```cpp
void matmul_q8(
    ap_int<8> A[N * N],                           // Q8_0 weight bytes (4096 B)
    combined_scale_t combined_scales[N * 2],      // 128 × UQ8.8 fixed-point (256 B)
    in_t X[N],                                     // Activation vector (INT16, 128 B)
    acc_t Y[N],                                    // Output vector (INT64, 512 B)
    volatile ap_uint<32> *control,                 // AXI-Lite control register pointer
    volatile ap_uint<32> *status,                  // AXI-Lite status register pointer
    ap_uint<1> &interrupt                          // Interrupt output line
) {
```

The first 4 arguments are AXI master (m_axi) ports — the ARM writes data to these DDR addresses before starting the kernel. The last 3 are AXI-Lite slave (s_axilite) — registers accessed via AXI-Lite.

### Lines 246-253: HLS interface pragmas

```cpp
    #pragma HLS INTERFACE m_axi port=A bundle=aximm0 offset=slave depth=4096
    #pragma HLS INTERFACE m_axi port=combined_scales bundle=aximm1 offset=slave depth=128
    #pragma HLS INTERFACE m_axi port=X bundle=aximm2 offset=slave depth=64
    #pragma HLS INTERFACE m_axi port=Y bundle=aximm3 offset=slave depth=64
    #pragma HLS INTERFACE s_axilite port=control bundle=control
    #pragma HLS INTERFACE s_axilite port=status bundle=control
    #pragma HLS INTERFACE ap_vld port=interrupt bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control
```

**4 separate AXI master ports** (bundle=aximm0..3): Each is an independent AXI-MM interface connected to a separate High-Performance (HP) port on the Zynq PS. This allows parallel DDR reads for A, combined_scales, and X, plus a write for Y, all overlapping.

**`offset=slave`**: Address offsets are controlled via AXI-Lite registers — the ARM writes the base DDR address for each port.

**`depth=4096` etc**: Tells HLS the maximum number of data transfers, used to size internal address counters.

**`bundle=control`**: The control, status, return, and interrupt all share one AXI-Lite slave interface (one set of registers accessed by ARM via GP0 port).

### Line 255: BRAM resource hint

```cpp
    #pragma HLS RESOURCE variable=A core=RAM_2P_URAM
```

**NOTE**: This should be `RAM_2P_BRAM` for Zynq 7010. `RAM_2P_URAM` is for Ultrascale devices that have URAM blocks. Zynq 7010 only has BRAM. This line may cause an error or be silently ignored by Vitis HLS targeting 7-series devices.

### Lines 257-269: Control handshake

```cpp
    *status = STATUS_IDLE;          // signal: ready

    bool start = (*control) & 0x1;  // read bit 0: ARM wrote 1 to start
    if (!start) {
        return;                     // not started — return immediately
    }

    *status = STATUS_RUNNING;        // signal: computing

    ap_uint<1> int_enable = ((*control) >> 1) & 0x1;  // bit 1: interrupt enable
    ap_uint<1> op_vecmul = ((*control) >> 3) & 0x1;   // bit 3: 0=matmul, 1=vecmul

    bool vec_mode = (op_vecmul == 1);
```

ARM writes to control register:
- bit 0 = 1 → start
- bit 1 = 1 → enable interrupt on completion
- bit 3 = 1 → vecmul mode (decode), 0 → matmul mode (prefill)

The FPGA reads these bits on the first cycle after start is asserted.

### Lines 271-297: Vecmul path (DECODE)

```cpp
    if (vec_mode) {
        ap_int<8> B_mat[N][N];              // local BRAM for Q8 weights
        #pragma HLS ARRAY_PARTITION variable=B_mat complete

        // Load Q8 weight bytes from DDR
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                #pragma HLS PIPELINE II=1
                B_mat[i][j] = A[i * N + j];    // A is flat 4096-byte DDR buffer
            }
```

Loads the 64×64 = 4096 INT8 bytes from DDR into local BRAM. The AXI master port reads contiguous bytes from DDR starting at the address programmed by ARM.

```cpp
        // Load combined scales from DDR
        combined_scale_t cs_mat[N][2];
        #pragma HLS ARRAY_PARTITION variable=cs_mat complete dim=1
        for (int i = 0; i < N; i++)
            for (int b = 0; b < 2; b++) {
                #pragma HLS PIPELINE II=1
                cs_mat[i][b] = combined_scales[i * 2 + b];
            }
```

128 UQ8.8 values loaded from DDR into 64×2 local register array.

```cpp
        acc_t result[N];
        vecmul_1x64_q8(B_mat, cs_mat, X, result);    // compute

        // Write results to DDR
        for (int i = 0; i < N; i++) {
            #pragma HLS PIPELINE II=1
            Y[i] = result[i];
        }
```

### Lines 298-337: Matmul path (PREFILL)

```cpp
    } else {
        in_t B_int16[N][N];                 // local BRAM for dequantized weights
        #pragma HLS ARRAY_PARTITION variable=B_int16 complete

        // Same scale load as vecmul path
        combined_scale_t cs_mat[N][2];
        ...

        // Dequant: Q8 → INT16
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                #pragma HLS PIPELINE II=1
                int block_idx = j / Q8_BLOCK_SIZE;
                int offset = j % Q8_BLOCK_SIZE;
                ap_int<8> q8_val = A[i * N + Q8_BLOCK_BYTES * block_idx + 2 + offset];
                B_int16[i][j] = q8_dequant_lut(q8_val, cs_mat[i][block_idx]);
            }
        }
```

**BUG (same as vecmul path)**: The index `Q8_BLOCK_BYTES * block_idx + 2 + offset` = `34 * (j/32) + 2 + (j%32)`. For j=0: 2. For j=31: 33. For j=32: 36. For j=63: 67. This is beyond the 64-element column and assumes Q8 block structure in the flat array.

The correct index is simply `j` (or equivalently `A[i * N + j]`), since the flat array contains only raw INT8 values.

```cpp
        // Create activation matrix: row 0 = X, rows 1..63 = 0
        in_t X_mat[N][N];
        #pragma HLS ARRAY_PARTITION variable=X_mat complete
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                #pragma HLS PIPELINE II=1
                X_mat[i][j] = (i == 0) ? X[j] : 0;
            }

        acc_t Y_mat[N][N];
        matmul_64x64(X_mat, B_int16, Y_mat);

        // Extract first row as output
        for (int i = 0; i < N; i++) {
            #pragma HLS PIPELINE II=1
            Y[i] = Y_mat[0][i];
        }
    }
```

For the prefill path, we broadcast a single activation vector as row 0 of the A matrix and run the full 64×64 matmul. The resulting row 0 of C = X @ B, which is the desired 1×64 output.

### Lines 339-342: Completion handshake

```cpp
    *status = STATUS_DONE;           // signal: done
    if (int_enable) {
        interrupt = 1;               // assert interrupt line
    }
}
```

The ARM either polls `status` register (reads until STATUS_DONE) or configures the interrupt controller to handle the done signal. The `interrupt` line stays high until the ARM writes to the ISR (interrupt status register) via AXI-Lite, which clears it in the top-level testbench.

---

## Performance Summary

| Mode | Cycles/tile | MACs/tile | MACs/cycle | DSPs |
|------|------------|-----------|-----------|------|
| Vecmul (decode) | 64 + 4096 load + 64 write ≈ 4288 | 4096 | 64 | 64 |
| Matmul (prefill) | ~1500 load + ~1152 compute + 4096 write ≈ 6750 | 4096 | ~0.6 average | 64 |

**Vecmul**: 64 parallel MACs per cycle (fully utilized). For a 151936×896 logits matmul: 33,236 tiles × 64 cycles = 2.13M compute cycles @ 150 MHz = 14.2 µs. Total with AXI transfers: ~175 µs (dominated by 4096 bytes/tile × 33,236 tiles / 600 MB/s ≈ 225 µs DDR bandwidth).

**Key numbers for the 8×8 systolic array**:
- 64 DSP48E1 running at 150 MHz
- Each DSP: INT16×INT16→INT48 in one cycle (actually INT17×INT17)
- Peak: 64 MAC/cycle × 150 MHz = 9.6 GMAC/s
- Utilization: each tile = 4096 MACs in 64 cycles = 64 MACs/cycle = 100% DSP utilization during compute
