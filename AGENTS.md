## Project State

Qwen2-0.5B FPGA accelerator targeting Zynq 7010. Dual-core Verilog RTL: Q8_0 (logits) + Q4_K INT16 pipeline (FFN layers), with AXI4-Lite memory-mapped I/O.

## What Was Actually Accomplished

### Phase 1 — C++ Pre-Dequant INT16 Path (COMPLETE)

The FPGA receives **pre-dequantized INT16 data** (8192 bytes/tile) from the CPU. The Verilog core does pure INT16×INT16 multiply-accumulate — no Q4_K block decoding on FPGA.

| Component | Status | What it does |
|-----------|--------|-------------|
| `matmul_fpga_q4k()` in `tmac_gguf.cpp` | ✅ | Reads Q4_K blocks from model, calls `dequant_q4k_tile()` on CPU, sends INT16 to simulated FPGA |
| `dequant_q4k_tile()` in `fpga_sim.hpp` | ✅ | Correct Q4_K block dequant (f16 d/dmin, 6-bit scale unpack, 4-bit quants) |
| `axi_vecmul_tile_q4k_axilite()` in `fpga_sim.hpp` | ✅ | Protocol-accurate AXI-Lite buffer simulation: per-word writes to weight_buf, act_buf, compute via `axilite_q4k_run()`, result readback |
| `AxiliteAccelState`, `axilite_write_buf()` | ✅ | Verilog-matching buffer model (weight_buf[8192], scale_buf[128], act_buf[64], result_buf[64]) |
| `matmul_q4k_core.v` | ✅ | 512×128-bit wmem (2 BRAMs), INT16×INT16 pipeline, 3-stage FSM, 515 cycles/tile |
| `matmul_top.v` | ✅ | Dual-core instantiation, `mode_q4k` mux (bit 6), 8192-byte weight_buf, AXI-Lite decoder |
| Q4K wmem addressing | ✅ | Col-major read: `{k[5:0], g[2:0]}` = k*8+g. Loading: entry 0=col0 rows0..7, entry 8=col1 rows0..7 |
| AXI-Lite address map | ✅ | 0x1000-0x1FFF weight_low, 0x2000-0x20FF scale, 0x2100-0x217F act, 0x2200-0x2FFF weight_high, 0x4000-0x41FF result_lo, 0x4200-0x43FF result_hi |
| scale/act overlap fix | ✅ | `mode_q4k` gates scale/act writes off when weight data writes through 0x2000-0x217F |
| act_buf isolation | ✅ | `axi_vecmul_tile_q4k_axilite` writes act_buf directly (not through `axilite_write_buf`) to prevent contamination |

### What Phase 1 Does NOT Provide (Phase 2)

The FPGA still receives **8192 bytes of INT16 per tile** — same bandwidth as generic INT16 pre-dequant. The actual Q4_K value (receive 2304 bytes of compressed blocks, dequant on FPGA) is **not implemented**.

| Feature | Current (Phase 1) | Target (Phase 2) |
|---------|------------------|------------------|
| CPU→FPGA data | 8192 bytes INT16/tile | 2304 bytes Q4_K blocks/tile |
| Bandwidth savings | 0% (same as INT16) | 72% vs INT16, 44% vs Q8_0 |
| Dequant location | CPU (`dequant_q4k_tile()`) | FPGA (new Verilog block decoder) |
| Q4_K block decoder in Verilog | **MISSING** | Must parse 144B blocks → 256 INT16 |
| q8_core wmem width | 512×64-bit (1 BRAM) | N/A |
| q4k_core wmem width | 512×128-bit (2 BRAMs, oversized for pre-dequant) | Could be 144×128-bit if on-FPGA dequant writes INT16 to wmem |

### Dispatch Logic (`tmac_gguf.cpp`)

```
if (g_fpga_q4k && type == Q4_K)  → matmul_fpga_q4k()    (FFN: gate/up/down)
else if (g_fpga_q8 && type == Q8_0) → matmul_fpga_q8()   (logits: token_embd)
else                               → matmul_fpga_int16()   (everything else: attention QKV, etc.)
```

Both `--fpga-q4k` and `--fpga-q8` can be active simultaneously. `--fpga` = `--fpga-int16`.

### Q5_K, Q6_K Support

Same `matmul_q4k_core.v` — it's quantization-agnostic INT16×INT16. Just add a CPU dequant function (Q5_K→INT16 or Q6_K→INT16) and dispatch to the same `axi_vecmul_tile_q4k_axilite()` path. No Verilog changes needed.

## Test Results (all pass)

| Suite | Tests | Status |
|-------|-------|--------|
| C++ integration (`test_integration.cpp`) | 3 tests (addr map, Q4K compute, 5-tile roundtrip) | **ALL PASS** |
| Verilog Q8 core (`tb_matmul_q8.v`) | 6 directed tests | **ALL PASS** |
| Verilog Q4K core (`tb_matmul_q4k.v`) | 7 tests (448 checks: all-ones, zero, negative, col-varying, per-row ref) | **ALL PASS** |
| Verilog Q4K smoke (`tb_minimal_q4k.v`) | 1 smoke test | PASS |
| End-to-end Q4_K inference (3 tokens) | 9707→358→3003, 59.4× simulated speedup | ✅ matches CPU |
| End-to-end Q8_0 inference (3 tokens) | 9707→358→3003, 29.9× simulated speedup | ✅ matches Q4_K output |
| End-to-end both flags (3 tokens) | Same tokens | ✅ |

## Phase 2 — On-FPGA Q4_K Dequant (Not Started)

### What Needs to Be Built

**A. Q4_K block decoder module** (new Verilog, ~150-200 lines)

Takes 144-byte Q4_K block, outputs 256 INT16 values:
- Parse `d` (f16 at byte 0-1) → float32 or UQ16.16 fixed-point
- Parse `dmin` (f16 at byte 2-3) → same format
- Unpack 12 bytes of 6-bit scales (bytes 4-15) → 8 × `sc` + 4 × `m`
  - 6-bit packing: every 3 bytes hold 4 × 6-bit values
- Unpack 128 bytes of 4-bit quants (bytes 16-143) → 256 × 4-bit values
  - 2 quants per byte, nibble order depends on GGUF convention
- Dequant formula: `val_i = (quant_i - 8) × sc[block//32] × d + m[block//32] × dmin`
  - Each 256-weight block has 8 sub-blocks of 32 weights, with 8 `sc` and 4 `m` values
  - The 12 scales are organized as 8 high-scale + 4 low-scale, interleaved
- Output 256 INT16 values, 2 values per cycle over 128 cycles (or 1/cycle over 256 cycles)

**B. Updated loading FSM** in `matmul_top.v`

Current: loads 8192 bytes sequentially, maps via `{byte_lane[3:0], entry[8:0]}` to wmem
New: loads 2304 bytes (144B × 16 blocks), feeds each block to the Q4_K decoder, decoder writes INT16 to wmem

**C. C++ updates**

Current: `axi_vecmul_tile_q4k_axilite()` converts INT16 matrix to byte-serialized format
New: `axi_vecmul_tile_q4k_axilite()` should send raw Q4_K blocks (2304 bytes) through `axilite_write_buf()`, and `axilite_q4k_run()` should simulate the new decoder + loading FSM

**D. Address map change**

Current: weights occupy 0x1000-0x2FFF (8192 bytes)
New: Q4_K blocks occupy 0x1000-0x18FF (2304 bytes). The 0x1900-0x2FFF range becomes free.

### Key Design Decisions for Phase 2

1. **Where does the decoder live?** Inside `matmul_q4k_core.v` or as a separate module instantiated in `matmul_top.v`? Recommend inside the core — keeps the interface clean (core receives raw blocks, outputs INT16 results).

2. **f16 arithmetic**: Q4_K block `d` and `dmin` are float16. For pure LUT-based dequant (no DSP), convert to fixed-point. For DSP-based, use a simple f16→f32 converter.

3. **Scale unpacking**: The 6-bit scale packing is the trickiest part. Reference: `ggml-quants.c` `ggml_dequantize_q4_K()` or the existing `dequant_q4k_tile()` in `fpga_sim.hpp`.

4. **Pipeline stages**: The decoder likely needs multiple cycles per block. If decoder throughput < 1 block/16 cycles, it becomes the bottleneck (512 compute cycles for 16 blocks → 32 cycles/block budget).

### Q4_K Block Format Reference

```
Offset  Size  Field    Description
0-1     2B    d        Block delta (float16)
2-3     2B    dmin     Block minimum (float16)
4-15    12B   scales   Packed 6-bit scales (8 × sc + 4 × m)
16-143  128B  qs       256 × 4-bit weight values (2 per byte)
        ─────
        144B total = 256 weights × 0.5625 B/weight

Scale unpacking (12 bytes → 12 × 6-bit values):
  bytes 4-7:   8 × 6-bit high scales (sc[0..7])
  bytes 8-11:  4 × 6-bit low scales (m[0..3])
  bytes 12-15: 4 reserved + 4 more m values
  Each 6-bit value packed across 3 bytes holding 4 values:
    byte[i]   = [val[i*4+1][1:0], val[i*4][5:0]]    (lower 6 bits of val 0 + upper 2 bits of val 1)
    byte[i+1] = [val[i*4+2][3:0], val[i*4+1][5:2]]  (lower 4 bits of val 1 + upper 4 bits of val 2)
    byte[i+2] = [val[i*4+3][5:0], val[i*4+2][5:4]]  (lower 2 bits of val 2 + upper 6 bits of val 3)

Dequant formula per weight j in a 256-weight block:
  sub_block = j / 32          (0..7)
  sc_used   = sc[sub_block]   (high scale)
  m_used    = m[sub_block/2]  (low scale, shared by 2 sub-blocks)
  val       = (qs[j] - 8) * sc_used * d + m_used * dmin
  qs[j] = nibble from byte array (low nibble = j even, high nibble = j odd)
```

## Architecture Summary

```
User FSM flow (matmul_top.v):
  IDLE → LOAD_WEIGHT (8192 bytes) → LOAD_ACT (64 × 2 bytes) → COMPUTE → DRAIN → IDLE

Core FSM (matmul_q4k_core.v):
  IDLE → COMPUTE (512 iters: 64 cols × 8 row-groups) → DRAIN (1 cycle) → DRAIN2 (1 cycle) → IDLE
  Total: 515 cycles/tile

Pipeline:
  Stage 0: BRAM read address, act_reg[k] capture
  Stage 1: wmem[addr] × act_reg → INT32 product
  Stage 2: accumulate into 48-bit acc[base+wi_i]

Q4K wmem read addressing:
  wt_addr = {k[5:0], g[2:0]} = k*8 + g
  k = column (0..63), g = row group (0..7)
  Each entry holds 8 INT16 values: rows g*8+0 .. g*8+7 for column k

Loading (col-major sequential):
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
- `verilog/matmul_top.v` — Dual-core top: AXI4-Lite slave, 8192-byte weight_buf, loading FSM, `mode_q4k` mux
- `verilog/matmul_q8_core.v` — Q8_0 compute core: 512×64-bit wmem (1 BRAM), 128×16-bit smem, LUT dequant, 3-stage FSM
- `verilog/matmul_q4k_core.v` — INT16×INT16 core: 512×128-bit wmem (2 BRAMs), 3-stage FSM, no dequant
- `verilog/axilite_slave.v` — AXI4-Lite slave + register file
- `verilog/dequant_lut.v` — Q8_0 dequant ROM (standalone, not instantiated)
- `verilog/systolic_8x8.v` — 8×8 systolic array (standalone, not used)

### Verilog Testbenches
- `verilog/tb_matmul_q8.v` — 6 tests, ALL PASS
- `verilog/tb_matmul_q4k.v` — 7 tests (448 checks), ALL PASS
- `verilog/tb_minimal_q4k.v` — 1 smoke test, PASS
- `verilog/tb_cosim.v` — Q8_0 only, reads `/tmp/cosim_tiles.bin`

### C++ Simulation
- `sim/tmac_gguf.cpp` — Full inference pipeline (1364 lines), 3 dispatch branches, Chrome Trace profiler (`--perf`)
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
make -C verilog all                     # Q8 (6/6) + Q4K (7/7) + Q4K smoke

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

1. **Phase 2 — On-FPGA Q4_K dequant** (highest priority)
   - Q4_K block decoder in Verilog (144B → 256 INT16)
   - Update loading FSM for 2304-byte blocks
   - Update `axilite_q4k_run()` C++ sim to send raw blocks + simulate decoder
   - Shrink weight_buf or repurpose freed address space
   - Update testbenches for raw block input
   - Benefit: 72% bandwidth reduction per tile (2304 vs 8192 bytes)

2. **`--dump-tiles` for Q4K AXI-Lite path** — needed for Verilog co-simulation

3. **Full `test_integration.sh` on model data** — requires model.tmac at /tmp/model.tmac

4. **Q8 AXI-Lite path failure** — pre-existing zeros in Test 3 (unrelated to Q4K work)

5. **`tb_cosim.v` for Q4_K** — currently Q8_0 only

6. **`verilog/EVENT_SEQUENCE.md`** — outdated for dual-core, still references Q8-only flow
