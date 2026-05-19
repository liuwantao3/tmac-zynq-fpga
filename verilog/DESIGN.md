# Verilog Q8_0 Accelerator — Design Document

## Overview

Custom Verilog RTL for Q8_0 vector-matrix multiplication on Xilinx Zynq 7010, replacing the HLS-based approach. Uses a 3-stage pipeline with 8 parallel dequant+MAC lanes and BRAM-based weight storage.

## Status

| Component | Status |
|-----------|--------|
| Core compute (`matmul_q8_core.v`) | Complete — 3-stage pipeline, 2 optimizations applied |
| Dequant LUT (`dequant_lut.v`) | Complete — Q8_0: INT8 × UQ8.8 → INT16 |
| Systolic array (`systolic_8x8.v`) | Complete — standalone, not instantiated |
| Top-level (`matmul_q8_top.v`) | Complete — AXI4-Lite + BRAM buffers |
| AXI-Lite slave (`axilite_slave.v`) | Complete — register file |
| Core testbench (`tb_matmul_q8.v`) | Complete — 6/6 tests pass |
| Co-simulation (`tb_cosim.v`) | Complete — 5 tiles (320 checks) pass |

---

## Architecture

### Memory Organization

```
Weight storage (wmem): 512 × 64-bit = 4096 bytes → 1 BRAM36
  addr = bank * 64 + col = {bank[2:0], col[5:0]}
  data = 8 packed Q8_0 bytes: {row7, row6, ..., row0}

Scale storage (smem): 128 × 16-bit = 256 bytes → distributed RAM
  addr = row * 2 + block (block=0 for cols 0..31, block=1 for cols 32..63)

Activation storage (act_reg): 64 × 16-bit = 128 bytes → distributed RAM

Accumulator (acc): 64 × 48-bit = 384 bytes → distributed RAM/FF
```

### 3-Stage Pipeline

```
Stage 0 (address): BRAM read address {g, k} set, act_reg[k] + smem[...] read
Stage 1 (compute):  BRAM data arrives + dequant + multiply by activation → partial
Stage 2 (accumulate): partial products added to accumulator
```

**Cycle breakdown per tile (64 cols × 8 banks = 512 iterations):**
- IDLE exit: 1 cycle
- COMPUTE × 512: pipeline fill + compute
- DRAIN: 1 cycle (drain Stage 1 partial)
- DRAIN2: 1 cycle (drain Stage 2 partial)
- **Total: 515 cycles/tile**

### Datapath

```
                act_reg[k]  smem[sc_addr]
                     │           │
                     ▼           ▼
BRAM addr {g,k} → wmem_rdata → dequant_q8(w, s) → × act → acc[base+offset]
```

Each COMPUTE iteration processes one column (k) and one bank group (g=0..7):
- 8 weights (rows g*8+0 through g*8+7) at column k
- 8 scales (one per row)
- 1 activation value (same for all 8 lanes)
- Produces 8 partial products → accumulates into acc[g*8+0..g*8+7]

---

## Optimizations Applied

### Optimization 1: BRAM-based Weight Storage

**Before:** 3D array `reg [7:0] wmem [0:7][0:63][0:7]` — inferred as distributed RAM (LUTs)

**After:** Flat 1D array `reg [63:0] wmem [0:511]` with `(* ram_style = "block" *)` — inferred as single BRAM36

**Impact:**
- Frees 2-3K LUTs used by large distributed RAM mux
- BRAM read is synchronous (1-cycle latency), extending pipeline from 2 to 3 stages
- 64-bit read returns all 8 weights in one cycle

**Byte-write handling:**
```verilog
always @(posedge clk) begin
    if (wt_we) begin
        case (wt_addr[8:6])  // byte lane
            3'd0: wmem[{wt_addr[11:9], wt_addr[5:0]}][7:0]   <= wt_din;
            3'd1: wmem[{wt_addr[11:9], wt_addr[5:0]}][15:8]  <= wt_din;
            // ... lanes 2-7
        endcase
    end
end
```

### Optimization 2: Single-bank Accumulators

**Before:** Double-buffer `acc` / `acc_nxt` with cycle-by-cycle copy

**After:** Single `reg signed [47:0] acc [0:63]` with read-before-write:
```verilog
acc[base + wi_i] <= acc[base + wi_i] + partial[wi_i];
```

**Impact:**
- Saves ~3K FFs (copy loop eliminated)
- No separate copy always block
- Reset clears acc directly

---

## Modules

### `matmul_q8_core.v` — Compute Core

**I/O:**
| Signal | Width | Direction | Description |
|--------|-------|-----------|-------------|
| clk | 1 | in | Clock |
| rst_n | 1 | in | Active-low reset |
| start | 1 | in | Start computation |
| op_vecmul | 1 | in | 0=MatMul, 1=VecMul |
| done | 1 | out | Computation complete |
| busy | 1 | out | Currently computing |
| wt_we | 1 | in | Weight write enable |
| wt_addr | 12 | in | Weight write address |
| wt_din | 8 | in | Weight write data |
| sc_we | 1 | in | Scale write enable |
| sc_addr | 7 | in | Scale write address |
| sc_din | 16 | in | Scale write data |
| act_we | 1 | in | Activation write enable |
| act_addr | 6 | in | Activation write address |
| act_din | 16 | in | Activation write data |
| res_addr | 6 | in | Result read address |
| res_dout | 48 | out | Result read data |

**State machine:** IDLE → COMPUTE (×512) → DRAIN → DRAIN2 → IDLE

**Dequantization:**
```verilog
function automatic signed [15:0] dequant_q8;
    input signed [7:0]  q8;   // INT8 weight
    input [15:0] sc;         // UQ8.8 scale
    // Returns: q8 * scale >> 8 = q8 * (scale / 256)
```

### `dequant_lut.v` — Q8_0 Dequant LUT

Standalone module for Q8_0 dequantization. Not instantiated in current design (function-based dequant inside `matmul_q8_core.v`).

### `systolic_8x8.v` — 8×8 Systolic Array

Standalone 8×8 INT16 systolic array. Not currently used — the actual compute uses 8 parallel direct MAC lanes, not systolic array.

### `matmul_q8_top.v` — Top-level Integration

Top module with:
- AXI4-Lite control interface
- Internal BRAM data buffers (weights, scales, activations)
- Interrupt output
- Instantiation of `matmul_q8_core`

### `axilite_slave.v` — AXI4-Lite Slave

AXI4-Lite slave with register file for control/status.

---

## Testbenches

### `tb_matmul_q8.v` — Core Testbench

6 tests, all passing:

| Test | Weights | Scales | Acts | Expected |
|------|---------|--------|------|----------|
| 1 | 1 | 1.0 | 1 | 64 |
| 2 | 0 | 1.0 | 1 | 0 |
| 3 | 1 | 2.0 | 1 | 128 |
| 4 | 1 | 1.0 | 2 | 128 |
| 5 | 1 | 1.0 | 0 | 0 |
| 6 | -1 | 1.0 | 1 | -64 |

**Run:**
```bash
cd verilog && make clean && make sim
```

### `tb_cosim.v` — Co-simulation Testbench

Reads real model tile data from `/tmp/cosim_tiles.bin` (generated by `tmac_gguf --dump-tiles N`).

**Binary tile dump format:**
```
Header (16 bytes):
  [num_tiles: u32] [reserved: u32 × 3]

Per tile (4992 bytes):
  [q8_W: 4096 bytes]     — col-major: q8_tile[col][row]
  [scales: 256 bytes]    — 128 × u16 (UQ8.8)
  [vec: 128 bytes]       — 64 × i16
  [expected: 512 bytes] — 64 × i64 (fpga_sim reference)
```

**Run:**
```bash
echo "1" | ./sim/tmac_gguf /path/to/model.tmac --fpga-q8 --dump-tiles 5
cd verilog && make cosim
```

**Results:** 5 tiles, 320 checks — all PASS

---

## Co-simulation Workflow

```
1. C++ engine dumps tiles:
   echo "1" | ./sim/tmac_gguf model.tmac --fpga-q8 --dump-tiles 5
   → /tmp/cosim_tiles.bin

2. Verilog testbench reads and computes:
   cd verilog && make cosim
   → 320 result checks PASS
```

---

## Build & Simulation

```bash
# Basic tests
cd verilog && make sim

# Co-simulation (requires tiles)
make cosim

# View waveforms
make waves        # basic
make cosim-waves # cosim

# Clean
make clean
```

**Tools:** iverilog + vvp (Icarus Verilog, `brew install icarus-verilog`)

---

## Resource Estimation (Post-synthesis)

| Resource | Estimated | Available | Usage |
|----------|-----------|-----------|-------|
| DSP | 8 | 80 | 10% |
| BRAM | 1-3 (36Kb each) | 60 | 2-5% |
| LUT | 2.5-3.5K | 17,600 | 14-20% |
| FF | 6.5-9.7K | 35,200 | 37-55% |

*Note: Only Vivado synthesis provides exact numbers. iVerilog simulation uses behavioral models.*

---

## Performance

| Metric | Value |
|--------|-------|
| Cycles per tile | 515 |
| Tiles per token | ~120,596 |
| Total cycles | ~62M |
| Time @ 150 MHz | ~413 ms/token |
| Speedup vs CPU | ~47× |

*Actual performance depends on DDR bandwidth and AXI interconnect overhead.*

---

## History

| Date | Change |
|------|--------|
| 2026-05-19 | Two optimizations applied: BRAM weights (flat 1D array), single-bank accumulators. Pipeline extended to 3 stages. 515 cycles/tile confirmed. |
| 2026-05-19 | Three bugs fixed in original `matmul_q8_core.v`: weight write address swap, shared `a_i` variable, shared `wi_int` variable. |
| 2026-05-19 | Co-simulation framework created (`--dump-tiles` flag, `tb_cosim.v`). 5 real-model tiles verified. |