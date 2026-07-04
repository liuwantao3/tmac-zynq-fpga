# Verilog Accelerator — Design Document

## Overview

Custom Verilog RTL for 5 quantization formats on Xilinx Zynq 7010:
- Q8_0: token embedding, attn_v, logits (151936×896)
- Q5_0: attn_q/k/o, ffn_gate/up (896×896, 4864×896)
- Q6_K: ffn_down even layers (896×4864)
- Q4_K: ffn_down odd layers (896×4864)
- INT16: F32 norm fallback

## Status (2026-07-05)

| Component | Status |
|-----------|--------|
| Q8_0 core (`matmul_q8_core.v`) | Complete — 6-stage pipeline, 8×BRAM18, 524 cycles/tile, 6/6 tests PASS |
| Q4_K core (`matmul_q4k_core.v`) | Complete — 56×256 tile, 4/4 tests PASS |
| Q5_0 core (`matmul_q5_0_core.v`) | Complete — 8×896 tile, 32/32 tests PASS |
| Q6_K core (`matmul_q6_k_core.v`) | Complete — 32×256 tile, 97/97 tests PASS |
| INT16 core (`matmul_int16_core.v`) | Complete — 64×64 tile, smoke test (addressing issue) |
| HP FSM Top (`hp_fsm_top.v`) | Complete — AXI4-Lite + HP read/write + Q8 compute, 9/9 HW tests PASS |

## Architecture

### 6-Stage Pipeline (Q8 core)

```
PRE → Stage 0 → Stage 1a → Stage 1b → Stage 2a → Stage 2b
```

| Stage | Operation |
|-------|-----------|
| PRE | Register BRAM read addresses (g, k, wmem_addr, smem_addr, act_addr) |
| Stage 0 | BRAM read data arrives: wmem[8B], smem[8×16b], act[16b] |
| Stage 1a | Dequant: INT8 weight × UQ8.8 scale → INT16 (function `dequant_q8`) |
| Stage 1b | Multiply: INT16 dequant × INT16 act → S24.8 partial |
| Stage 2a | Pre-read accumulator BRAM for next cycle's write-back |
| Stage 2b | Accumulate: acc_bank[addr] += partial (read-modify-write BRAM) |

**Memory:**
- wmem_bank0..7: 8× BRAM18, each 512×8 (one byte lane per bank)
- smem_bank0..7: 8× BRAM18, each 512×16 (scale storage)
- act_bram: 1× BRAM18, 512×16 (activation vector)
- acc_b0..b7: 8× BRAM18, each 512×48 (accumulator banks)

**Cycle breakdown (64×64 sub-tile):**
- CLEAR_ACC: 8 cycles (bulk zero all BRAM acc banks)
- COMPUTE: 512 cycles (64k × 8g = 512 MAC iterations, 8 MACs/cycle)
- DRAIN: 4 cycles (pipeline flush = DRAIN..DRAIN4)
- **Total: 524 cycles/tile**

### Q4_K Core (56×256 tile)

Block buffer decode. Decodes Q4_K blocks into S24.8 fixed-point values, accumulates row-wise. No wmem (streaming block decode). Cycles: 14,337/tile (56 rows × 256 cols, one element/cycle).

### Q5_0 Core (8×896 tile)

Serial block decode (2 cycles/element). 224 blocks of 32 elements each = 7,168 elements. Cycles: 14,338/tile.

### Q6_K Core (32×256 tile)

Block decode with 32 blocks, super_scale + per-sub-block scales. Cycles: 8,194/tile.

### INT16 Core (64×64 tile)

3-stage pipeline (same pattern as original Q8). 512×128-bit wmem. Cycles: 515/tile.

### HP FSM (hp_fsm_top.v)

20-state FSM with descriptor-chain DMA. See docs/AGENTS.md for register map and state table.

## Resource Utilization (post-implementation, 2026-07-05)

| Resource | Used | Available | % |
|----------|------|-----------|--|
| Slice LUTs | 7,769 | 17,600 | 44.1 |
| LUT as Logic | 7,710 | 17,600 | 43.8 |
| LUT as Memory | 59 | 6,000 | 0.98 |
| Slice Regs | 12,293 | 35,200 | 34.9 |
| BRAM18 | 17 | 120 | 14.2 |
| DSP48E1 | 16 | 80 | 20.0 |
| WNS | 0.601 ns | 10 ns | 100 MHz |

## Testbenches

| Testbench | Module | Tests | Status |
|-----------|--------|-------|--------|
| `tb_matmul_q8.v` | `matmul_q8_core` | 6 directed | ALL PASS |
| `tb_matmul_q4k.v` | `matmul_q4k_core` | 4 tests | ALL PASS |
| `tb_matmul_q5_0.v` | `matmul_q5_0_core` | 32 tests | ALL PASS |
| `tb_matmul_q6_k.v` | `matmul_q6_k_core` | 97 tests | ALL PASS |
| `tb_int16_smoke.v` | `matmul_int16_core` | 1 smoke | PASS (known wmem addr issue) |
| `tb_cosim.v` | Q8 co-sim | 5 tiles | PASS |
| `tb_cosim_q4k.v`, `tb_cosim_q5_0.v`, `tb_cosim_q6_k.v` | Co-sim w/ `dump-tiles` | — | PASS |
| `tb_hw_fsm_comprehensive.v` | HP FSM | 7+2 tests | ALL PASS |
| **Hardware** | HP FSM on Zynq | 9 tests | **ALL PASS** |

```bash
cd verilog && make all           # all core tests
make -C verilog sim_q8           # Q8 only
make -C verilog sim_q4k          # Q4K only
make -C verilog sim_q5_0         # Q5_0 only
make -C verilog sim_q6_k         # Q6_K only
```
