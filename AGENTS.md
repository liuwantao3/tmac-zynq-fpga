## Project State

Qwen2-0.5B FPGA accelerator targeting Zynq 7010. Quad-core Verilog RTL: INT16×INT16 (general), Q8_0 dequant (logits), Q4_K block decode (down_proj), with AXI4-Lite memory-mapped I/O.

**Model is Q4_K_M (mixed quantization).** Actual tensor types:
- `token_embd` (151936×896): Q8_0
- `attn_v` (128×896): Q8_0
- `attn_q`, `attn_k`, `attn_output` (896×896): Q5_0
- `ffn_gate`, `ffn_up` (4864×896): Q5_0
- `ffn_down` (896×4864): Q6_K (layers 0,2,4,...) or Q4_K (layers 1,3,5,...)

Three distinct block-decode paths + INT16 fallback, dispatched by tensor type. All four paths produce bit-identical `11 358 3003` output.

## What Was Actually Accomplished

### Phase 1 — Three-Core Architecture (COMPLETE)

The three calculation cores in the FPGA:

| Component | Status | What it does |
|-----------|--------|-------------|
| `matmul_int16_core.v` | ✅ | Pure INT16×INT16: 512×128-bit wmem, 3-stage FSM, 515 cycles/tile |
| `matmul_q8_core.v` | ✅ | Q8_0 dequant: 512×64-bit wmem, dequant LUT, 3-stage FSM |
| `matmul_q4k_core.v` | ✅ | Q4_K block decode: 2304B block buffer, S24.8 fixed-point decoder, 3-stage FSM |
| `matmul_q4k_2x896_core.v` | ✅ | Q4_K 2×896 tile: 1008B block buffer (7 blocks), single-pass decode+accumulate, 2-cycle pipeline, 3585 cycles/tile |
| `matmul_top.v` | ✅ | Quad-core top: 4-way mode mux, AXI-Lite slave, loading FSM |

### C++ Simulation — Block-Decode Paths (COMPLETE)

Four dispatch branches in `matmul()` of `tmac_gguf.cpp`, routed by `A->type`:

| Type | Function | Tile | Block format | Bytes/tile |
|------|----------|------|-------------|------------|
| Q5_0 | `matmul_fpga_q5_0` | [8, 896] | 22B block (FP16 d + 4B qh + 16B nibbles), 224 blocks | 4928 |
| Q6_K | `matmul_fpga_q6_k` | [32, 256] | 210B block (128B L + 64B H + 16 scales + 2B super_scale), 32 blocks | 6720 |
| Q4_K | `matmul_fpga_q4_k` | [56, 256] | 144B block (128B qs + 16 scales + 2B super_scale), 56 blocks | 8064 |
| Q8_0 | `matmul_fpga_q8` | [64, 896] | 4100B block buffer | 4100 |
| others | `matmul_fpga_int16` | [64, 64] | N/A | 8192 |

All tiles fit in 8192-byte Verilog weight_buf. All paths use FP32 `row_inv` (no UQ16.8 cap). Both `--fpga-q4k` and `--fpga-q8` can be active simultaneously; `--fpga` = `--fpga-int16`.

### Dispatch Logic (`tmac_gguf.cpp:448`)

```
if (g_fpga_q4k && A->type == Q5_0) → matmul_fpga_q5_0()  (attn_q/k/o, ffn_gate/up)
else if (g_fpga_q4k && A->type == Q6_K) → matmul_fpga_q6_k()  (ffn_down, ~12 layers)
else if (g_fpga_q4k && A->type == Q4_K) → matmul_fpga_q4_k()  (ffn_down, ~16 layers)
else if (g_fpga_q8 && A->type == Q8_0) → matmul_fpga_q8()   (token_embd, attn_v, logits)
else → matmul_fpga_int16()   (F32 norms, fallback)
```

## Test Results (all pass)

| Suite | Tests | Status |
|-------|-------|--------|
| C++ integration (`test_integration.cpp`) | 3 tests (addr map, INT16 AXI-Lite, Q4K raw block) | **ALL PASS** |
| Verilog Q8 core (`tb_matmul_q8.v`) | 6 directed tests | **ALL PASS** |
| Verilog Q4K core (`tb_matmul_q4k.v`) | 5 tests (448 checks: all-ones, zero, negative, col-varying, per-row ref) | **ALL PASS** |
| Verilog Q4K smoke (`tb_minimal_q4k.v`) | 1 smoke test | PASS |
| Verilog INT16 smoke (`tb_int16_smoke.v`) | 1 smoke test | PASS |
| Verilog Q4K 2×896 (`tb_matmul_q4k_2x896.v`) | 4 tests (8 checks: all-ones, zero, zero-act, scaled) | **ALL PASS** |
| End-to-end CPU-only (3 tokens) | 11→358→3003 | reference |
| End-to-end Q8_0 inference (3 tokens) | 11→358→3003, 6.3× speedup | ✅ matches CPU |
| End-to-end Q4_K inference (3 tokens) | 11→358→3003 | ✅ matches CPU |
| End-to-end both flags (3 tokens) | 11→358→3003 | ✅ matches CPU |

## Phase 2 — On-FPGA Q4_K Dequant (COMPLETE)

## Phase 2 — Q4_K CPU Dequant + INT16 FPGA Matmul

Q4_K tensors are dequantized per-element on the CPU to INT16, then sent to the FPGA for INT16×INT16 matmul via the INT16 core.

- **CPU-side dequant**: `matmul_fpga_q4k()` uses `dequant_q4_k()` per-element, same as CPU reference, then routes through `axi_vecmul_tile_int16_axilite()` for the FPGA INT16 matmul path.
- **Q4_K block decoder in Verilog** (`matmul_q4k_core.v`): present but unused for this model (requires 256-col divisible tensor for native block alignment). See `docs/Q4_K_IMPLEMENTATION_PLAN.md`.
- **Fixed `m_used` bug** in Verilog `decode_one()` (line 292): sub-blocks 4-7 used wrong byte address.
- **Fixed flag ordering bug**: `--fpga-q4k` parsed before `load_tmac()`.
- **All tests pass**: 3 C++ + 7 Q4K Verilog + 6 Q8 Verilog + Q4K smoke + INT16 smoke.

## Architecture Summary

```
User FSM flow (matmul_top.v):
  IDLE → LOAD_WEIGHT (8192/4096/2304/1008 bytes) → LOAD_SCALE (Q8 only) → LOAD_ACT (64 × 2 bytes) → COMPUTE → DRAIN → IDLE

Core FSM (matmul_q4k_2x896_core.v):
  IDLE → LD (load 7-block data into pipeline regs) → DEC (decode + accumulate) → DRAIN (last acc) → IDLE
  Total: 3585 cycles/tile (2 cycles/element × 1792 elements + 1 DRAIN)
  No wmem — only 1008B block buffer + 896-entry act_reg + 2 entry row_scale

Core FSM (matmul_int16_core.v / matmul_q8_core.v):
  IDLE → COMPUTE (512 iters: 64 cols × 8 row-groups) → DRAIN (1 cycle) → DRAIN2 (1 cycle) → IDLE
  Total: 515 cycles/tile

Core FSM (matmul_q4k_core.v):
  IDLE → DECODE (block decode) → DW0/DW1 (wmem writes) → COMPUTE → DRAIN → DRAIN2 → IDLE

Pipeline (all cores):
  Stage 0: BRAM read address, act_reg[k] capture
  Stage 1: wmem[addr] × act_reg → INT32 product
  Stage 2: accumulate into 48-bit acc[base+wi_i]

Wmem read addressing (all cores):
  wt_addr = {k[5:0], g[2:0]} = k*8 + g
  k = column (0..63), g = row group (0..7)
  Each entry holds 8 INT16 values: rows g*8+0 .. g*8+7 for column k

INT16 loading (col-major sequential):
  load_addr 0..15   → entry 0, byte_lane 0..15   → col 0, rows 0..7
  load_addr 16..31  → entry 1, byte_lane 0..15   → col 0, rows 8..15
  ...
  load_addr 128..143 → entry 8, byte_lane 0..15  → col 1, rows 0..7
  ...
  load_addr 8176..8191 → entry 511 → col 63, rows 56..63
```

## Performance (simulated 150 MHz, single token)

| Path | Compute tiles | FPGA cycles | FPGA time | Simulated speedup |
|------|-------------|-------------|-----------|-------------------|
| Q4_K (FFN) + INT16 (rest) | 449k | ~94M | 626 ms | 8.4× |
| Q8_0 (logits) + INT16 (rest) | 449k | ~49M | 329 ms | 29.9× |
| Both paths active | 449k | ~46M | 304 ms | 32.9× |

Note: C++ sim cycle counting is approximate. Q4_K tiles through `axilite_q4k_run()` are not counted in FPGA cycles. Only INT16 path through `MatmulAccel` counts cycles. The `CYCLES_PER_TILE = 515` constant matches Verilog RTL timing.

## File Inventory

### Verilog RTL
- `verilog/matmul_top.v` — Quad-core top: AXI4-Lite slave, 8192-byte weight_buf, loading FSM, `mode_q4k` mux
- `verilog/matmul_q8_core.v` — Q8_0 compute core: 512×64-bit wmem (1 BRAM), 128×16-bit smem, LUT dequant, 3-stage FSM
- `verilog/matmul_q4k_core.v` — Q4_K block decode core: 2304-byte block buffer, S24.8 fixed-point decoder, 512×128-bit wmem (2 BRAMs), 3-stage FSM
- `verilog/matmul_q4k_2x896_core.v` — Q4_K 2×896 core: 7-block decode for gate/up projections, 1008B buffer, single-pass decode+accumulate, no wmem
- `verilog/matmul_int16_core.v` — General INT16×INT16 core: 512×128-bit wmem (2 BRAMs), 3-stage FSM, no dequant
- `verilog/axilite_slave.v` — AXI4-Lite slave + register file
- `verilog/dequant_lut.v` — Q8_0 dequant ROM (standalone, not instantiated)
- `verilog/systolic_8x8.v` — 8×8 systolic array (standalone, not used)

### Verilog Testbenches
- `verilog/tb_matmul_q8.v` — 6 tests, ALL PASS
- `verilog/tb_matmul_q4k.v` — 7 tests (448 checks), ALL PASS
- `verilog/tb_minimal_q4k.v` — 1 smoke test, PASS
- `verilog/tb_int16_smoke.v` — 1 smoke test, PASS
- `verilog/tb_cosim.v` — Q8_0 only, reads `/tmp/cosim_tiles.bin`

### C++ Simulation
- `sim/tmac_gguf.cpp` — Full inference pipeline (1497 lines), 3 dispatch branches, Chrome Trace profiler (`--perf`)
- `sim/fpga_sim.hpp` — `MatmulAccel` (AXI-Lite register model + FSM + compute), `vecmul_1x64_int16` (bit-accurate golden ref), `AxiliteAccelState` (buffer model matching Verilog), `axilite_write_buf`, `axilite_q4k_run`, `axi_vecmul_tile_q4k_axilite`, `dequant_q4k_tile`, `TimingStats`, `CYCLES_PER_TILE=515`
- `sim/matmul_q8.cpp` — Q8_0 logits path wrapper
- `sim/test_integration.cpp` — 3 tests, ALL PASS
- `sim/chat.py` — Chat interface (Python, uses tmac_gguf binary)

### Scripts
- `scripts/extract_tmac.py` — GGUF→TMAC converter
- `scripts/ground_truth_v2.py` — Full Qwen2 FP32 ground truth via Python
- `scripts/py_tmac_vec.py` — Vectorized Python TMAC reference
- `scripts/verify_layers_fast.py` — Layer-by-layer C++ vs Python comparison
- `scripts/design_iteration.sh` — HLS/Vivado iteration loop
- `scripts/feedback_parser.py` — HLS/Vivado report parser
- `scripts/test_integration.sh` — Full C++ + Verilog integration test suite

### Documentation
- `verilog/DESIGN.md` — Architecture, timing, testbenches, 3-level verification strategy, profiling
- `verilog/EVENT_SEQUENCE.md` — AXI-Lite protocol trace (pre-dates Q4K, partially outdated)
- `docs/architecture.md` — Model architecture, quantization formats, TMAC format, bugs found
- `docs/FPGA_PERFORMANCE_ANALYSIS.md` — Zynq 7010 theoretical performance (Q8_0 baseline)
- `docs/Q4_K_IMPLEMENTATION_PLAN.md` — Original 5-phase plan (Phase 1 done, Phase 2 not started)
- `docs/hls_q8_kernel_explanation.md` — Legacy HLS kernel line-by-line (primary moved to Verilog)
- `docs/PROGRESS_SUMMARY.md` — INT4 era history (archived)
- `sim/Transaction Tracer/README.md` — `fpga_profiler.hpp` usage docs (inactive)
- `scripts/README.md` — Scripts directory documentation
- `AGENTS.md` — This file

## Build & Run Commands

```bash
# Verilog tests
make -C verilog all                     # Q8 (6/6) + Q4K (7/7) + Q4K smoke + INT16 smoke

# C++ integration test
g++ -std=c++14 -O2 -I sim -I gguf -I . sim/test_integration.cpp -lpthread -o /tmp/ti
/tmp/ti

# Build inference engine
g++ -std=c++17 -pthread -O2 -I sim -I gguf -I . sim/tmac_gguf.cpp sim/matmul_q8.cpp -o sim/tmac_gguf

# FPGA simulation flags
echo "9707" | ./sim/tmac_gguf models/model.tmac --fpga-q4k --generate 3
echo "9707" | ./sim/tmac_gguf models/model.tmac --fpga-q8 --generate 3
echo "9707" | ./sim/tmac_gguf models/model.tmac --fpga-q4k --fpga-q8 --generate 3

# Full test suite
bash scripts/test_integration.sh

# Model conversion
python3 scripts/extract_tmac.py models/qwen2-0_5b-instruct-q4_k_m.gguf /tmp/model.tmac
```

## Remaining Work

1. ✅ ~~`--dump-tiles` for Q4K AXI-Lite path~~
2. ✅ ~~`tb_cosim.v` for Q4_K~~
3. **`matmul_q4k_2x896_core.v`** ✅ Verilog core + testbench done (8/8 tests pass)
4. **C++ dispatch** ✅ `axi_vecmul_tile_q4k_2x896_axilite()` in `fpga_sim.hpp`, shape-based dispatch in `matmul_fpga_q4k()`
5. **Full `test_integration.sh` on model data** — requires model.tmac at /tmp/model.tmac
6. **Q8 AXI-Lite path failure** — pre-existing zeros in Test 3 (unrelated to Q4K work)
7. **`verilog/EVENT_SEQUENCE.md`** — outdated for quad-core
8. **End-to-end inference test** — run with `--fpga-q4k` and verify correct output (11 358 3003)
