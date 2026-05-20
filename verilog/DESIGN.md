# Verilog Accelerator — Design Document

## Overview

Custom Verilog RTL for Q8_0 and Q4_K vector-matrix multiplication on Xilinx Zynq 7010.
Dual-core architecture with mode switching: Q8_0 for token embedding (logits), Q4_K for FFN layers (pre-dequantized INT16 data), INT16 for attention QKV.

## Status

| Component | Status |
|-----------|--------|
| Q8_0 core (`matmul_q8_core.v`) | Complete — 3-stage pipeline, 6/6 testbench tests pass |
| Q4_K core (`matmul_q4k_core.v`) | Complete — INT16×INT16 pipeline, 512×128-bit wmem (2 BRAMs), 7/7 tests pass |
| Top-level (`matmul_top.v`) | Complete — dual-core instantiation, `mode_q4k` muxing, 8192-byte weight_buf |
| AXI-Lite slave (`axilite_slave.v`) | Complete — register file |
| Core testbench Q8 (`tb_matmul_q8.v`) | Complete — 6/6 tests pass |
| Core testbench Q4K (`tb_matmul_q4k.v`) | Complete — 7/7 tests pass (448/448 checks) |
| Co-simulation (`tb_cosim.v`) | Q8_0 only — reads `/tmp/cosim_tiles.bin` |

---

## Architecture

### Dual-Core Topology

```
┌─────────────────────────────────────────────────────────────┐
│                      matmul_top.v                            │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  AXI4-Lite Write Decoder → weight_buf[8192]            │ │
│  │                         → scale_buf[128]               │ │
│  │                         → act_buf[64]                  │ │
│  └─────────────────────────────────────────────────────────┘ │
│                            │                                  │
│              ┌─────────────┴─────────────┐                    │
│              ▼                            ▼                    │
│  ┌──────────────────────┐   ┌──────────────────────┐          │
│  │ matmul_q8_core       │   │ matmul_q4k_core      │          │
│  │ 512×64-bit wmem (1B) │   │ 512×128-bit wmem (2B)│          │
│  │ Q8→INT16 dequant     │   │ INT16×INT16 pipeline  │          │
│  │ smem[128] scales     │   │ (no scale/act overlap)│          │
│  │ 3-stage FSM          │   │ 3-stage FSM           │          │
│  └──────────────────────┘   └──────────────────────┘          │
│              │                            │                    │
│              └─────────────┬─────────────┘                    │
│                            ▼                                  │
│              ┌──────────────────────┐                         │
│              │ mode_q4k MUX         │                         │
│              │ (bit 6 of ctrl_user) │                         │
│              └──────────────────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

### Memory Organization

**Q8_0 Core:**
- Weight storage (wmem): 512 × 64-bit = 4096 bytes → 1 BRAM36
  - `addr = bank * 64 + col = {bank[2:0], col[5:0]}`
  - `data = 8 packed Q8_0 bytes`
- Scale storage (smem): 128 × 16-bit = 256 bytes → distributed RAM
  - `addr = row * 2 + block`
- Activation storage (act_reg): 64 × 16-bit → distributed RAM

**Q4_K Core:**
- Weight storage (wmem): 512 × 128-bit = 8192 bytes → 2 BRAM36
  - `addr = {k[5:0], g[2:0]}` (col-major: k=column, g=group-of-8-rows)
  - `data = 8 pre-dequantized INT16 values`
- No scale storage (pre-dequant on CPU)
- Activation storage (act_reg): 64 × 16-bit → distributed RAM

### 3-Stage Pipeline (shared pattern)

```
Stage 0 (addr):  BRAM read address issued, act_reg[k] captured
Stage 1 (mul):   BRAM data arrives × act_reg → INT32 partial
Stage 2 (acc):   partial products accumulated into 48-bit acc
```

**Cycle breakdown per tile (64 cols × 8 groups = 512 iterations):**
- IDLE exit: 1 cycle
- COMPUTE × 512
- DRAIN: 1 cycle (flush Stage 1)
- DRAIN2: 1 cycle (flush Stage 2)
- **Total: 515 cycles/tile** (both cores)

### Q4_K Wmem Address Mapping

The Q4_K core reads wmem with `{k, g}` addressing:

- `k` = column (0..63), 6 bits
- `g` = group of 8 rows (0..7), 3 bits
- wmem address = `k * 8 + g` = 9 bits (0..511)

Loading writes sequentially from tile data in col-major order:
- Entry 0 → col 0, rows 0..7
- Entry 1 → col 0, rows 8..15
- ...
- Entry 8 → col 1, rows 0..7
- ...
- Entry 511 → col 63, rows 56..63

---

## Modules

### `matmul_q8_core.v` — Q8_0 Compute Core

**I/O:** 1 clock, start/done/busy handshake, wt/sc/act write ports, 48-bit result read.

**State machine:** IDLE → COMPUTE (×512) → DRAIN → DRAIN2 → IDLE (515 cycles/tile)

**Datapath:**
```
act_reg[k]  smem[sc_addr]
     │           │
     ▼           ▼
wmem → dequant_q8(w, s) → × act → acc[base+wi_i]
```

**Dequantization:**
```verilog
dequant = q8_val * scale >> 8  // INT8 × UQ8.8 → INT16
```

### `matmul_q4k_core.v` — Q4_K Compute Core

**I/O:** Same interface as Q8 core except:
- `wt_addr`: 13 bits (vs 12 bits for Q8) — wider for 512-entry × 128-bit wmem
- No `sc_we` / `sc_addr` / `sc_din` (pre-dequant on CPU)

**State machine:** Same 3-stage FSM pattern (IDLE → COMPUTE → DRAIN → DRAIN2)

**Datapath:**
```
wmem → INT16 weight × INT16 act → acc[base+wi_i]
```

### `matmul_top.v` — Top-level Integration

Dual-core AXI-Lite wrapper:
- AXI4-Lite write decoder routes to weight_buf / scale_buf / act_buf
- Loading FSM copies buffers to selected core's internal memories
- Mode select (`reg_ctrl_user[6]`): 0 = Q8_0, 1 = Q4_K
- In Q4K mode: scale/act writes disabled at overlapping addresses (0x2000-0x217F)
- 8192-byte weight_buf (was 4096 for Q8-only)

**AXI-Lite address map:**
| Range | Buffer | Bytes |
|-------|--------|-------|
| 0x0000-0x00FF | Control registers | 256 |
| 0x1000-0x1FFF | Weight buffer low | 4096 |
| 0x2000-0x20FF | Scale buffer (Q8 only) | 256 |
| 0x2100-0x217F | Activation buffer | 128 |
| 0x2200-0x2FFF | Weight buffer high (Q4K) | 3584 |
| 0x4000-0x41FF | Result buffer low 32b | 256 |
| 0x4200-0x43FF | Result buffer high 16b | 256 |

### `axilite_slave.v` — AXI4-Lite Slave

Register file: AP_CTRL, GIE, IER, ISR, CTRL_USER, STATUS.

### `dequant_lut.v` — Q8_0 Dequant LUT

Standalone ROM for Q8_0 dequantization. Not instantiated in current design (function-based dequant inside `matmul_q8_core.v`).

### `systolic_8x8.v` — 8×8 Systolic Array

Standalone 8×8 INT16 systolic array. Not currently used — the actual compute uses 8 parallel direct MAC lanes.

---

## Testbenches

### `tb_matmul_q8.v` — Q8_0 Core Testbench

6 directed tests, all passing:

| Test | Weights | Scales | Acts | Expected |
|------|---------|--------|------|----------|
| 1 | 1 | 1.0 | 1 | 64 |
| 2 | 0 | 1.0 | 1 | 0 |
| 3 | 1 | 2.0 | 1 | 128 |
| 4 | 1 | 1.0 | 2 | 128 |
| 5 | 1 | 1.0 | 0 | 0 |
| 6 | -1 | 1.0 | 1 | -64 |

```bash
cd verilog && make sim_q8
```

### `tb_matmul_q4k.v` — Q4_K Core Testbench

7 directed tests, all passing (448/448 checks):

| Test | Weights | Acts | Expected |
|------|---------|------|----------|
| 1 | all 1 | all 1 | 64 |
| 2 | all 0 | all 1 | 0 |
| 3 | all 1 | all 2 | 128 |
| 4 | all 1 | all 0 | 0 |
| 5 | all -1 | all 1 | -64 |
| 6 | col-varying (10+col) | all 1 | 2656 |
| 7 | mixed per-row ref | mixed per-row ref | dot_product per row |

```bash
cd verilog && make sim_q4k
```

### `tb_minimal_q4k.v` — Q4_K Smoke Test

Quick regression test: 64 active columns all with weight=INT16(1), act=1, verify row 0 = 64.

```bash
cd verilog && make sim_q4k_min
```

### `tb_cosim.v` — Q8_0 Co-simulation

Reads real model tile data from `/tmp/cosim_tiles.bin` (generated by `tmac_gguf --dump-tiles N`).
Q8_0 only — 5 tiles, 320 checks verified passing.

**Binary tile dump format (4992 bytes/tile):**
```
Header (16 B): [num_tiles: u32] [reserved: u32 × 3]
Per tile: [q8_W: 4096 B] [scales: 256 B] [vec: 128 B] [expected: 512 B]
```

```bash
echo "1" | ./sim/tmac_gguf model.tmac --fpga-q8 --dump-tiles 5
cd verilog && make sim_cosim
```

---

## Build & Simulation

```bash
# Q8_0 core tests
cd verilog && make sim_q8

# Q4_K core tests
make sim_q4k

# Q4_K smoke test
make sim_q4k_min

# Co-simulation (requires tiles)
make sim_cosim

# All core tests
make all

# View waveforms
make waves-q8
make waves-q4k

# Clean generated artifacts
make clean
```

**Tools:** iverilog + vvp (Icarus Verilog, `brew install icarus-verilog`)

---

## Resource Estimation (Post-synthesis)

| Resource | Q8_0 Core | Q4_K Core | Total | Available | Usage |
|----------|-----------|-----------|-------|-----------|-------|
| DSP | 0 | 8 | 8 | 80 | 10% |
| BRAM36 | 1 | 2 | 3 | 60 | 5% |
| LUT | ~2.5K | ~1K | ~3.5K | 17,600 | ~20% |
| FF | ~6.5K | ~3.2K | ~9.7K | 35,200 | ~28% |

*Q4_K core uses BRAM for wmem, distributed RAM for accumulators, no DSP (LUT-based INT16×INT16 multiply).*

---

## Performance

| Metric | Q8_0 | Q4_K | INT16 (CPU pre-dequant) |
|--------|------|------|-------------------------|
| Cycles/tile | 515 | 515 | 515 |
| BW per tile | 4 KB + 256 B | 8 KB | 8 KB |
| Sim FPGA time/token | 182 µs | 127 µs | 146 µs |
| Speedup vs CPU | 37.5× | 60.4× | 49.9× |

*FPGA time measured at simulated 150 MHz. INT16 path includes CPU pre-dequant overhead.*

---

---
## Verification & Simulation Strategy

Three levels of verification ensure correctness from algorithmic behavior through RTL timing:

### Level 1: C++ Functional Simulation

Validates dequantization, INT16×INT16 matmul arithmetic, and AXI-Lite buffer protocol — all in software.

**Files:**
- `sim/tmac_gguf.cpp` — Full inference pipeline with FPGA backend dispatch (3 branches: Q4K→FFN, Q8→logits, else→INT16)
- `sim/fpga_sim.hpp` — `MatmulAccel` class (AXI-Lite register model + background FSM + compute), `vecmul_1x64_int16` (bit-accurate golden reference), `AxiliteAccelState` (buffer model matching Verilog), `axilite_write_buf`, `axilite_q4k_run`, `axi_vecmul_tile_q4k_axilite`
- `sim/test_integration.cpp` — 3 integration tests (address map, Q4K compute, 5-tile roundtrip) — ALL PASS

```bash
g++ -std=c++14 -O2 -I sim -I gguf -I . sim/test_integration.cpp -lpthread -o /tmp/test_integration
/tmp/test_integration
```

**Flags:** `--fpga` / `--fpga-int16`, `--fpga-q8`, `--fpga-q4k`, `--dump-tiles N`, `--dump-layers`, `--generate N`

### Level 2: Cycle-Budget Timing Model

Estimates FPGA performance: cycles/tile, total compute time, bandwidth, and CPU-vs-FPGA speedup.

**Components:**
- `fpga_sim.hpp::TileCycleBudget` — Static constants: `INT16_TILE_CYCLES=202`, `Q8_TILE_CYCLES=214`, `COMPUTE_ONLY_CYCLES=74`, `US_PER_CYCLE=6.67ns`
- `fpga_sim.hpp::TimingStats` — Accumulates tile count, MAC operations, CPU wall time, FPGA cycle count. Reports speedup at end of inference.
- `MatmulAccel::run_compute()` — Accumulates `total_cycles_ = num_tiles × cycles_per_tile` after each compute.

Output at end of `tmac_gguf` run:
```
[FPGA TIMING SUMMARY]
  Tiles: N  MACs: M  CPU: X.XX ms  NAIVE: Y.YY Gop/s
  FPGA: Z cycles @ 150 MHz = W.WW us  Speedup: V.Vx
```

### Level 3: Verilog RTL Simulation (Icarus Verilog)

Validates actual RTL behavior with cycle-accurate testbenches. Four testbenches exist:

| Testbench | Module | Tests | Status |
|-----------|--------|-------|--------|
| `tb_matmul_q8.v` | `matmul_q8_core` | 6 directed tests | ALL PASS |
| `tb_matmul_q4k.v` | `matmul_q4k_core` | 7 tests (448 checks) | ALL PASS |
| `tb_minimal_q4k.v` | `matmul_q4k_core` | 1 smoke test | PASS |
| `tb_cosim.v` | `matmul_q8_core` | N tiles × 64 rows (Q8 only) | PASS (5 tiles verified) |

```bash
make all          # sim_q8 + sim_q4k + sim_q4k_min
make sim_cosim    # requires /tmp/cosim_tiles.bin
```

**Cosimulation flow:** `tmac_gguf --fpga-q8 --dump-tiles N` → writes `/tmp/cosim_tiles.bin` → `make sim_cosim` reads and checks against expected results.

---

## Performance Profiling Infrastructure

Two profiling systems coexist:

### 1. Chrome Trace Profiler (ACTIVE)

Built into `tmac_gguf.cpp`. Stack-based event tracing using `PROFILE_SCOPE(name)` macros.

**Activation:** `--perf` flag
**Output:** `/tmp/pipeline_trace.json` (Chrome trace viewer compatible) + terminal summary
**Usage:** 15+ `PROFILE_SCOPE` calls throughout inference pipeline (layer encode, RMS norm, Q4K matmul, Q8 matmul, attention, RoPE, etc.)

```cpp
PROFILE_SCOPE("matmul_fpga_q4k");  // auto begin+end via RAII
```

### 2. Pipeline Profiler (INACTIVE — retained for future use)

`sim/Transaction Tracer/fpga_profiler.hpp` — Standalone `Profiler` class tracking 6 pipeline stages:

| Stage | Value | Description |
|-------|-------|-------------|
| `STAGE_QUANTIZE` | 0 | Float → int8/int16 conversion |
| `STAGE_DDR_COPY` | 1 | memcpy to/from simulated DDR |
| `STAGE_AXI_SETUP` | 2 | AXI-Lite register writes |
| `STAGE_COMPUTE` | 3 | FPGA compute (simulated systolic array) |
| `STAGE_DEQUANTIZE` | 4 | INT32/INT64 → float + scale multiply |
| `STAGE_OVERHEAD` | 5 | Loop control, branches, etc. |

Features: nested timers (`Scope` RAII guard), per-tile / per-layer tracking, CSV export, bottleneck detection, optimization recommendations. Currently not instantiated anywhere; retained as a reference implementation for future coarser-grained profiling needs.

---

## History

| Date | Change |
|------|--------|
| 2026-05-19 | Initial Q8_0 core with 2 optimizations (BRAM weights, single-bank accumulators). 515 cycles/tile. Cosimulation framework. |
| 2026-05-20 | Q4_K core added: `matmul_q4k_core.v`, dual-core top `matmul_top.v`, 8192-byte weight_buf. Full test suite passes. |
| 2026-05-20 | Fixed wmem address mapping `{g,k}`→`{k,g}` to match sequential loading. All 7 Q4K tests pass. |
| 2026-05-20 | Dead code cleanup in `fpga_sim.hpp` (removed DspPhase, PipelineStats, AxiTiming, 6 orphaned functions). Removed stale `#include "Transaction Tracer/fpga_profiler.hpp"`. Added 3-level verification documentation. |
