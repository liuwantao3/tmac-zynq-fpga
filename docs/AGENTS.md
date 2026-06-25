# Project State

## Qwen2-0.5B FPGA Accelerator — Zynq 7010

Quad-core Verilog RTL: INT16×INT16 (general), Q8_0 dequant (logits), Q5_0 block decode (attn_q/k/o, ffn_gate/up), Q6_K block decode (ffn_down even), Q4_K block decode (ffn_down odd), with AXI4-Lite memory-mapped I/O.

**Model**: Q4_K_M (mixed quantization). See AGENTS.md top-level for tensor types table.

## Architecture

### User FSM flow (hp_fsm_top.v):
```
IDLE → FETCH_DESC → LOAD_ACT → WRITE_RES → DONE → (restart IDLE or FETCH_DESC)
```
Multi-burst HP reads: each burst capped at 64 bytes (AXI3: 16 beats × 32-bit ARSIZE=2).

### CPU-OP Descriptor Protocol
Descriptor type 15 (tensor_type=15) triggers CPU interrupt. FSM pauses in `PH_CPU_OP_WAIT` until CPU completes RMSNorm/RoPE/SoftMax/SwiGLU and resumes.

### Descriptor Chain (32 bytes each, DDR)
| Offset | Field | Size |
|--------|-------|------|
| 0x00 | next_addr | 4 |
| 0x04 | weight_addr | 4 |
| 0x08 | act_addr | 4 |
| 0x0C | result_addr | 4 |
| 0x10 | reserved | 4 |
| 0x14 | reserved | 4 |
| 0x18 | act_total_bytes | 4 |
| 0x1C | reserved | 4 |

## Register Map (hp_fsm_top, AXI4-Lite @ 0x43C00000)

| Offset | Name | Access | Description |
|--------|------|--------|-------------|
| 0x00 | reg_start | W | Bit[0] = start (auto-clears) |
| 0x14 | reg_status | R | [15]=busy, [9]=wr_done, [8]=rd_done |
| 0x18 | reg_desc_base | R/W | Descriptor base address |
| 0x1C | reg_desc_tail | R/W | Tail index |
| 0x20 | reg_desc_head | R | Head index (read-only) |
| 0x28 | reg_debug | R | [31:28]=state, [27]=rd_done, [26]=wr_done, [25]=rd_busy, [24]=wr_busy, [23:16]=act_remaining, [15:8]=rd_len, [7:0]=act_byte_idx |
| 0x2C | reg_clk_cnt | R | Free-running clock counter |
| 0x30 | reg_clk_cnt_slow | R | Clock counter ÷ 1024 |

### FSM states: 0=IDLE, 1=FETCH_DESC, 2=FETCH_DESC_W, 3=LOAD_ACT, 4=LOAD_ACT_W, 5=WRITE_RES, 6=WRITE_RES_W, 7=DONE

## RTL Changes (2026-06-21)

1. **DONE→FETCH_DESC restart** (`hp_fsm_top.v`): DONE state checks `reg_start` and transitions to FETCH_DESC, clearing `reg_status[15:8]`. Enables chain restart without bitstream reload.

2. **rd_len wrap guard** (`hp_fsm_top.v`): `(act_remaining >> 2) - 1` underflows when `act_remaining < 4` (produces rd_len=0xFF). Fixed with `>=4` / `>0` / `==0` branches. Prevents ARLEN=255 (PS7 AXI3 rejects >16 beats).

3. **Debug register always live**: `reg_debug[31:0]` continuously shows FSM state, rd/wr status, act_remaining, rd_len, byte_idx — not just latched.

## Key Hardware Findings

- **PS7 HP0 is AXI3**: max 16 beats per burst. ARLEN > 15 causes silent AR rejection (no ARREADY).
- **HP0 is 32-bit** on Zynq-7010 (x16 DDR3): `AFI0_CTRL[7:6]` (64-bit enable) is read-only. RDATA[63:32] always 0.
- **ps7_init hangs on re-run**: `ps7_pll_init_data_3_0` can't re-lock already-configured PLLs. Power-cycle required.
- **`rst -processor` corrupts DAP**: irreversibly until power-cycle. Always use `stop` instead.
- **FCLK_CLK0 enable**: DAP can't set `FPGA_CLK_CTRL[7]`. ARM boot code workaround (load to 0x00000000, wrpc, con, stop).

## Build Commands

```bash
# Proj build
C:\Xilinx\Vivado\2023.1\bin\vivado.bat -mode batch -source vivado_integration/build_bd.tcl

# Test run (after power-cycle)
C:\Xilinx\Vivado\2023.1\bin\xsdb.bat vivado_integration/sw/run_hp_fsm_test.tcl

# Verilog simulation (iVerilog — d:\iVerilog\bin)
make -C verilog all

# Vivado xsim (C:\Xilinx\Vivado\2023.1\bin\xsim.bat)
xsim tb_hw_fsm --runall
```

## Test Scripts (vivado_integration/sw/)

See `docs/test_infrastructure.md` for full catalog. Key scripts:
- `run_hp_loopback.tcl` — HP read+write loopback (proven on hw)
- `run_hp_fsm_test.tcl` — Descriptor chain FSM test
- `test_multiburst.tcl` — Multi-burst HP read test (in temp dir)

## File Inventory

### Verilog RTL
- `verilog/matmul_q8_core.v` — Q8_0 compute: 512×64 wmem, dequant LUT, 3-stage FSM
- `verilog/matmul_q4k_core.v` — Q4_K: 2304-byte block buffer, S24.8, 56×256 tile
- `verilog/matmul_q5_0_core.v` — Q5_0: 8×896 tile, 224 blocks, row_scale normalization
- `verilog/matmul_q6_k_core.v` — Q6_K: 32×256 tile, super_scale + sub-block scales
- `verilog/matmul_int16_core.v` — INT16×INT16: 512×128 wmem, 3-stage FSM
- `vivado_integration/rtl/hp_fsm_top.v` — Top FSM: AXI4-Lite, HP read/write masters, descriptor chain, multi-burst
- `vivado_integration/rtl/hp_loopback_top.v` — HP loopback test module

### C++ Simulation
- `sim/tmac_gguf.cpp` — Full inference pipeline, FPGA dispatch
- `sim/fpga_sim.hpp` — MatmulAccel, decode logic
- `sim/test_integration.cpp` — Integration tests

### Documentation
- `docs/test_infrastructure.md` — Register maps, script catalog, standardized flows
- `docs/debug_procedures.md` — XSDB reference, DAP error recovery, FSM debug
- `verilog/DESIGN.md` — Verilog architecture and timing
- `docs/architecture.md` — Model and quantization formats

## Pending

- **Verify multi-burst rd_len fix on hardware** — bitstream rebuilt, needs power-cycle + test
- **CPU boot crash (Prefetch Abort @ 0xFEBAB2B0)** — blocked until HP path verified
- End-to-end inference with all 5 compute cores
