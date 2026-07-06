# Project State

## Qwen2-0.5B FPGA Accelerator — Zynq 7010

Multi-core Verilog RTL: INT16×INT16 (general), Q8_0 dequant (logits), Q5_0 block decode (attn_q/k/o, ffn_gate/up), Q6_K block decode (ffn_down even), Q4_K block decode (ffn_down odd), with AXI4-Lite memory-mapped I/O.

**Model**: Q4_K_M (mixed quantization). See top-level AGENTS.md for tensor types table.

## Architecture

### FSM flow (hp_fsm_top.v, 29 states):

**Q8 compute path** (tensor_type=0):
```
IDLE → FETCH_DESC → FETCH_DESC_W → LOAD_WEIGHT → LOAD_WEIGHT_W →
  LOAD_SCALES → LOAD_SCALES_W → LOAD_ACT → LOAD_ACT_W →
  COPY_ACT_TO_CORE → COMPUTE → COMPUTE_W →
  [multi_group ? READ_RES_ACC(×groups) → COPY_ACC_TO_BUF : READ_RES] →
  WRITE_RES → WRITE_RES_BURST → WRITE_RES_W → DONE
```

**Q5_0 compute path** (tensor_type=1):
```
IDLE → FETCH_DESC → FETCH_DESC_W → Q5_PRELOAD_HDR → Q5_PRELOAD_HDR_W →
  Q5_LOAD_SCALES → Q5_LOAD_SCALES_W → Q5_COPY_ACT → Q5_COPY_ACT_W →
  loop(Q5_BLOCK_COMPUTE → Q5_BLOCK_COMPUTE_W ×56 blocks) → Q5_READ_RES →
  WRITE_RES → WRITE_RES_BURST → WRITE_RES_W → DONE
```

**CPU_OP path** (tensor_type=15): FETCH_DESC → LOAD_ACT → LOAD_ACT_W → WRITE_RES → DONE

### CPU-OP Descriptor Protocol
Descriptor type 15 (tensor_type=15) triggers CPU interrupt. FSM pauses until CPU completes RMSNorm/RoPE/SoftMax/SwiGLU and resumes via CHAIN_CTRL.

### Descriptor Format (32 bytes, DDR)

| Offset | Field | Size | Description |
|--------|-------|------|-------------|
| 0x00 | next_addr | 4 | Next descriptor DDR address (0 = end) |
| 0x04 | weight_addr | 4 | Q8 weight data address (4096 bytes) |
| 0x08 | act_addr | 4 | Activation data address |
| 0x0C | result_addr | 4 | Result writeback address |
| 0x10 | tensor_type | 2 | 15=CPU_OP, 0=Q8 compute, others reserved |
| 0x12 | (reserved) | 2 | |
| 0x14 | num_groups | 1 | Q8 column groups (0=use GP0 reg fallback) |
| 0x15 | (reserved) | 3 | |
| 0x18 | act_total_bytes | 4 | Total activation bytes to read |
| 0x1C | (reserved) | 4 | |

## Register Map (hp_fsm_top, AXI4-Lite @ 0x43C00000)

| Offset | Name | Access | Description |
|--------|------|--------|-------------|
| 0x00 | reg_start | W | Bit[0] = start (auto-clears) |
| 0x10 | reg_q8_num_groups | R/W | [3:0] column groups (0=single) |
| 0x14 | reg_status | R | [15]=busy, [9]=wr_done, [8]=rd_done |
| 0x18 | reg_desc_base | R/W | Descriptor base DDR address |
| 0x1C | reg_desc_tail | R/W | Tail index |
| 0x20 | reg_desc_head | R | Head index (read-only) |
| 0x28 | reg_debug | R | Debug status word (see below) |
| 0x2C | reg_clk_cnt | R | Free-running 100 MHz clock counter |
| 0x30 | reg_clk_cnt_slow | R | Clock counter ÷ 1024 |
| 0x34 | reg_act_info | R | act_addr from last descriptor |
| 0x38 | reg_desc_info | R | {8'h0, act_total_bytes[23:0]} |
| 0x3C | reg_q8_debug | R | Q8 core debug word |

### REG_DEBUG (0x28) bitfields (RTL lines 1255-1266):

| Bits | Field | Description |
|------|-------|-------------|
| [31:27] | state | FSM state (5-bit, 0-19) |
| [26] | rd_done | HP read master done (sticky) |
| [25] | wr_done | HP write master done (sticky) |
| [24] | rd_busy | HP read master busy |
| [23] | wr_busy | HP write master busy |
| [22] | q8_busy | Q8 core busy |
| [21:19] | wr_dbg_state | Write master FSM state (3-bit) |
| [18:16] | rd_dbg_state | Read master FSM state (3-bit) |
| [15] | q8_done | Q8 core done |
| [14:11] | col_group | Current column group (multi-group) |
| [10:8] | timeout_cnt[15:13] | Timeout counter upper bits |
| [7:0] | sc_byte_idx | Scale byte index |

### REG_Q8_DEBUG (0x3C) bitfields (RTL lines 1268-1277):

| Bits | Field | Description |
|------|-------|-------------|
| [31:27] | state | FSM state (5-bit) |
| [26] | q8_busy | Q8 core busy |
| [25] | q8_done | Q8 core done (pulse) |
| [24] | q8_start | Q8 core start (pulse) |
| [23] | q8_act_we | Q8 activation write enable |
| [22:20] | q8_core_state | Q8 core internal FSM state |
| [19:17] | q8_core_g | Q8 core bank counter |
| [16:11] | q8_core_k | Q8 core column counter |
| [10:7] | misc | {copy_act_idx[1:0], q8_sc_we, sc_byte_idx[0]} |
| [6:0] | wt_byte_idx[6:0] | Weight byte index |

### FSM States (REG_DEBUG[31:27]):

| Value | State | Description |
|-------|-------|-------------|
| 0 | IDLE | Waiting for reg_start |
| 1 | FETCH_DESC | Reading descriptor from DDR |
| 2 | FETCH_DESC_W | Draining descriptor read |
| 3 | LOAD_ACT | Starting HP read for activation data |
| 4 | LOAD_ACT_W | Draining act data into act_buf |
| 5 | WRITE_RES | Starting HP write for result |
| 6 | WRITE_RES_W | Draining HP write |
| 7 | DONE | Chain complete |
| 8 | LOAD_WEIGHT | Starting HP read for Q8 weights |
| 9 | LOAD_WEIGHT_W | Draining weight data into wmem |
| 10 | LOAD_SCALES | Starting HP read for scales |
| 11 | LOAD_SCALES_W | Draining + packing scales into smem |
| 12 | COPY_ACT_TO_CORE | Copying act_buf to Q8 act_reg |
| 13 | COMPUTE | Pulsing q8_start |
| 14 | COMPUTE_W | Waiting for Q8 core done |
| 15 | READ_RES | Reading Q8 results into act_buf (single-group) |
| 16 | READ_RES_ACC | Reading + accumulating (multi-group) |
| 17 | COPY_ACC_TO_BUF | Copying acc_buf to act_buf |
| 18 | TIMEOUT_ERROR | Timeout waiting for rd/wr/q8 done |
| 19 | WRITE_RES_BURST | Multi-burst result write |
| 20 | Q5_PRELOAD_HDR | Reading Q5_0 header data from DDR (672 bytes) |
| 21 | Q5_PRELOAD_HDR_W | Draining + unpacking headers to hdr_packed |
| 22 | Q5_LOAD_SCALES | Reading Q5_0 scale data from DDR |
| 23 | Q5_LOAD_SCALES_W | Draining + unpacking scales to row_scale |
| 24 | Q5_COPY_ACT | Reading Q5_0 activation data from DDR |
| 25 | Q5_COPY_ACT_W | Draining + unpacking acts to act_mem |
| 26 | Q5_BLOCK_COMPUTE | Per-block qs DDR read + clr_acc on blk=0 |
| 27 | Q5_BLOCK_COMPUTE_W | Assembling qs_word, pulsing start, waiting for done |
| 28 | Q5_READ_RES | Reading 4× rows from Q5_0 cores |

## Key Hardware Findings

- **PS7 HP0 is AXI3**: max 16 beats per burst. ARLEN > 15 causes silent AR rejection.
- **HP0 is 32-bit** on Zynq-7010 (x16 DDR3): RDATA[63:32] always 0.
- **ps7_init hangs on re-run**: PLLs can't re-lock if already configured. Power-cycle required.
- **`rst -processor` corrupts DAP**: irreversibly. Use `stop` instead.
- **FCLK_CLK0 enable**: DAP can't set FPGA_CLK_CTRL[7]. ARM boot code workaround required.

## Build Commands

```bash
# Vivado batch build
C:\Xilinx\Vivado\2023.1\bin\vivado.bat -mode batch -source vivado_integration/build_bd.tcl

# Hardware test (after power-cycle)
C:\Xilinx\Vivado\2023.1\bin\xsdb.bat vivado_integration/sw/run_hp_fsm_comprehensive.tcl

# Verilog simulation
make -C verilog all

# Vivado xsim
C:\Xilinx\Vivado\2023.1\bin\xsim.bat tb_hw_fsm --runall
```

## File Inventory

### Verilog RTL
- `verilog/matmul_q8_core.v` — Q8_0 compute: 8×BRAM18 wmem+banks, 6-stage pipeline, 524 cycles/tile
- `verilog/matmul_q4k_core.v` — Q4_K: 2304-byte block buffer, S24.8, 56×256 tile
- `verilog/matmul_q5_0_core.v` — Q5_0: 4×896 tile, 1 DSP MAC/cycle, block-streaming (SETUP_D→COMPUTE×32→DRAIN), 1849 cycles/tile
- `verilog/matmul_q6_k_core.v` — Q6_K: 32×256 tile, super_scale + sub-block scales
- `verilog/matmul_int16_core.v` — INT16×INT16: 512×128 wmem, 3-stage FSM
- `vivado_integration/rtl/hp_fsm_top.v` — Top FSM: AXI4-Lite, HP read/write masters, descriptor chain, Q8 compute integration

### C++ Simulation
- `sim/tmac_gguf.cpp` — Full inference pipeline, FPGA dispatch
- `sim/fpga_sim.hpp` — MatmulAccel, decode logic
- `sim/test_integration.cpp` — Integration tests

## Test Status (2026-07-06)

All HP FSM HW tests PASS: 7 baseline DMA + 2 Q8 (single/multi-group) + 9 Q5_0 dispatch.
Q8 core 6/6, Q4K 4/4, Q6_K 97/97. INT16 pre-existing failure.
Resource: 9,898 LUTs (56.2%), 33 BRAM18 (27.5%), 22 DSP (27.5%).
WNS = -9.74 ns at 100 MHz (works on HW despite violation).
