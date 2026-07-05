## Project State

Qwen2-0.5B FPGA accelerator targeting Zynq 7010. Quad-core Verilog RTL: INT16×INT16 (general), Q8_0 dequant (logits), Q5_0 block decode (attn_q/k/o, ffn_gate/up), Q6_K block decode (ffn_down even), Q4_K block decode (ffn_down odd), with AXI4-Lite memory-mapped I/O.

**Model is Q4_K_M (mixed quantization).** Actual tensor types:

| tensor | shape | type | C++ path | Verilog core |
|--------|-------|------|----------|-------------|
| `token_embd` | 151936×896 | Q8_0 | ✅ | ✅ `matmul_q8_core.v` |
| `attn_v` | 128×896 | Q8_0 | ✅ | ✅ `matmul_q8_core.v` |
| `attn_q`, `attn_k`, `attn_output` | 896×896 | Q5_0 | ✅ `matmul_fpga_q5_0` | ✅ `matmul_q5_0_core.v` |
| `ffn_gate`, `ffn_up` | 4864×896 | Q5_0 | ✅ `matmul_fpga_q5_0` | ✅ `matmul_q5_0_core.v` |
| `ffn_down` (layers 0,2,4,...) | 896×4864 | Q6_K | ✅ `matmul_fpga_q6_k` | ✅ `matmul_q6_k_core.v` |
| `ffn_down` (layers 1,3,5,...) | 896×4864 | Q4_K | ✅ `matmul_fpga_q4_k` | ✅ `matmul_q4k_core.v` (56×256 tile) |

## Key Decisions (2026-07-05)

1. **TB `wr` task: `input integer din` truncates 64-bit weight word** — Found: `tb_matmul_q8.v` declared `wr(input integer we, addr, din)` where `integer` is 32-bit signed. Passing `word = {8{val}}` (64-bit) truncated upper 32 bits, causing banks 4-7 to receive 0x00 for positive weights (all rows 4-7 = 0). Test 6 (wt=-1=0xFF) worked because sign-extension filled upper 32 bits with 1s. Fix: `input [63:0] din`. All 6 Q8 tests now PASS.

2. **Break statements removed from cosim testbenches** — `tb_cosim.v`, `tb_cosim_q4k.v`, `tb_cosim_q5_0.v`, `tb_cosim_q6_k.v` used unsupported `break` in Verilog for-loop wait loops. Replaced with `poll_count` flag loop condition.

3. **Q8 core: 64-bit word write, BRAM acc banks, dequant sat removed** — Q8 core rewritten:
   - Write port: `wt_addr[8:0]`/`wt_din[63:0]` replaces byte-lane BWE case (BRAM-friendly)
   - Accumulator: 8× BRAM18 banks (acc_b0..acc_b7), each 512×48, banked by address[2:0]=g
   - Dequant saturation removed: max product 127×65535=8,322,945 < 8,388,607, never saturates
   - Pipeline: 6-stage (PRE→S0→S1a→S1b→S2a→S2b), CLEAR_ACC state for bulk BRAM clear
   - Result: saves ~884 LUTs (384 LUTRAMs + 500 logic) vs old reg [47:0] acc[0:63]

4. **HP FSM: rd_ready <= rd_valid in LOAD_WEIGHT_W** — Fixed 0-cycle rvalid pulse: continuous `rd_ready=1` caused read master's PRESENT state to self-clear rvalid. Same delayed-handshake as LOAD_ACT_W.

5. **Multi-group Q8: q8_wt_din reg, col_group fix, act_remaining fix** — Three bugs from multi-group (2026-07-01): unregistered wt_din (NBA timing), col_group reset in COMPUTE_W, hardcoded act_remaining in READ_RES_ACC.

6. **All core unit tests PASS (2026-07-05):** Q8 6/6, Q4K 4/4, Q5_0 32/32, Q6_K 97/97, HP FSM 7/7. INT16 smoke pre-existing failure (unrelated wmem addressing).

7. **Track A BRAM conversion complete (2026-07-05):** All LUTRAM arrays converted to BRAM — 7,769 LUTs (44.1%), 17 BRAM18 (14.2%), 16 DSP (20%). WNS=0.601ns. All 9 HW tests PASS.

## Architecture Summary

### HP FSM flow (hp_fsm_top.v, Q8 compute path):
```
IDLE → FETCH_DESC → LOAD_WEIGHT → LOAD_SCALES → LOAD_ACT → COPY_ACT_TO_CORE → COMPUTE → READ_RES/READ_RES_ACC → WRITE_RES → DONE
```
For multi-group (q8_num_groups > 1, from descriptor): COMPUTE → READ_RES_ACC loops back to LOAD_SCALES for each group (accumulating into acc_buf), then COPY_ACC_TO_BUF → WRITE_RES.

### CPU-OP Descriptor Protocol (CPU/FPGA synchronization):

To handle CPU-only operations (RMSNorm, RoPE, SoftMax, bias add, SwiGLU, residual add) between FPGA matmuls, the descriptor chain supports a special `CPU_OP` descriptor type (`tensor_type = 15`).

When the FSM encounters a CPU_OP descriptor:
1. It sets `reg_chain_ctrl[2]=1` and pulses `desc_irq` → CPU interrupt fires
2. FSM enters `PH_CPU_OP_WAIT` state and **pauses** until CPU resumes it
3. CPU reads `reg_status` (status=3 = chain busy) to distinguish from chain-complete
4. CPU reads `reg_desc_head` to identify which descriptor index triggered the CPU_OP
5. CPU does its work (norm, softmax, etc.) and writes results to DDR
6. CPU clears `reg_isr[0]` (write REG_ISR) and writes `CHAIN_CTRL[0]=1` to resume
7. FSM clears the resume signal, advances to next descriptor

The CPU knows what operation to perform from the descriptor's position in the chain (the CPU built the chain, so it has an internal mapping: "descriptor 0 → attn_norm, descriptor 4 → bias+rope+softmax").

**CPU operations per layer** (from `tmac_gguf.cpp`):

| # | Operation | What it does | DDR I/O |
|---|-----------|-------------|---------|
| 1 | attn_norm | RMSNorm(hidden) | Read hidden, Write norm_out |
| 2 | q_bias | q += bias | Read q, Read bias, Write q |
| 3 | k_bias | k += bias | Read k, Read bias, Write k |
| 4 | v_bias | v += bias | Read v, Read bias, Write v |
| 5 | RoPE | apply_rope(q, k, pos) | Read q/k, Write q/k |
| 6 | SoftMax | score = q·k/√d, softmax, Σv·score | Read q/k/v + KV cache, Write context |
| 7 | residual | hidden += attn_out | Read hidden, Read attn_out, Write hidden |
| 8 | ffn_norm | RMSNorm(hidden) | Read hidden, Write norm2 |
| 9 | SwiGLU | silu(gate) * up | Read gate/up, Write swiglu_out |
| 10 | residual | hidden += ffn_out | Read hidden, Read ffn_out, Write hidden |

**Typical descriptor chain for one layer** (adjacent CPU ops batched):

```
Desc  0: CPU_OP           → attn_norm                          (result: norm_out)
Desc  1: matmul_q5_0      → attn_q     (act: norm_out)          (result: q)
Desc  2: matmul_q5_0      → attn_k     (act: norm_out)          (result: k)
Desc  3: matmul_q8_0      → attn_v     (act: norm_out)          (result: v)
Desc  4: CPU_OP           → bias+rope+softmax                   (result: context)
Desc  5: matmul_q5_0      → attn_output (act: context)          (result: attn_out)
Desc  6: CPU_OP           → residual+ffn_norm                   (result: norm2)
Desc  7: matmul_q5_0      → ffn_gate   (act: norm2)             (result: gate)
Desc  8: matmul_q5_0      → ffn_up     (act: norm2)             (result: up)
Desc  9: CPU_OP           → swiglu                              (result: swiglu_out)
Desc 10: matmul_q6k/q4k   → ffn_down   (act: swiglu_out)        (result: ffn_out)
Desc 11: CPU_OP           → residual                            (result: hidden)
```

12 descriptors × 28 layers = 336 descriptors per token. Each CPU_OP triggers one interrupt.
Adjacent CPU ops are batched into single descriptors (e.g. bias+rope+softmax = one CPU_OP).
Post-logits (softmax for sampling) is handled outside the descriptor chain.

### Descriptor Chain (OP→OP, no CPU):
Descriptors in DDR form a linked list. Each descriptor contains weight/act/result addresses, tensor type, and `act_total_bytes`. When descriptor N's `result_addr` equals descriptor N+1's `act_addr`, the chain auto-derives `act_total_bytes = prev.tile_res_rows × 8` (all cores output 48-bit fixed-point, 8 bytes/row). Header validation in `tb_phaseb.v` checks this invariant.

### Dispatch Logic (tmac_gguf.cpp:452-465):
```
if (g_fpga_q5_0 && A->type == TENSOR_Q5_0) → matmul_fpga_q5_0()  (attn_q/k/o, ffn_gate/up)
else if (g_fpga_q6_k && A->type == TENSOR_Q6_K) → matmul_fpga_q6_k()  (ffn_down even)
else if (g_fpga_q4k && A->type == TENSOR_Q4_K) → matmul_fpga_q4_k()  (ffn_down odd)
else if (g_fpga_q8 && A->type == TENSOR_Q8_0) → matmul_fpga_q8()   (token_embd, attn_v, logits)
else → matmul_fpga_int16()   (F32 norms, fallback)
```

### Tile Sizes and Buffer Usage:

| Type | Tile | Blocks | Bytes/tile | weight_buf | Result Bytes/row |
|------|------|--------|------------|------------|-----------------|
| Q8_0 | 64×896 | — | 4100 | 4096 | 8 |
| Q5_0 | 8×896 | 224 | 4928 | 8192 | 8 |
| Q6_K | 32×256 | 32 | 6720 | 8192 | 8 |
| Q4_K | 56×256 | 56 | 8064 | 8192 | 8 |
| INT16 | 64×64 | — | 8192 | 8192 | 8 |

All cores output S24.8 fixed-point (48-bit accumulator, zero-extended to 64-bit in DDR).

### HP FSM Register Map (AXI4-Lite @ 0x43C00000):

| Offset | Name | Access | Description |
|--------|------|--------|-------------|
| 0x00 | `REG_START` | R/W | [0]: write 1 to start descriptor chain (auto-clears when FSM leaves IDLE) |
| 0x14 | `REG_STATUS` | R/W | [8]=rd_done, [9]=wr_done, [15]=busy (cleared in IDLE/DONE) |
| 0x18 | `REG_DESC_BASE` | R/W | Descriptor base DDR address (32-bit) |
| 0x1C | `REG_DESC_TAIL` | R/W | Tail index (write 1 to enable chain, unused in current FSM) |
| 0x20 | `REG_DESC_HEAD` | R | Current descriptor index (read-only, increments after each WRITE_RES) |
| 0x28 | `REG_DEBUG` | R | Debug status word (see below) |
| 0x2C | `REG_CLK_CNT` | R | Free-running 32-bit clock cycle counter (increments every clk) |
| 0x30 | `REG_CLK_CNT_SLOW` | R | Clock counter divided by 1024 (for long timeouts) |
| 0x34 | `REG_ACT_INFO` | R | `act_addr` from last descriptor fetched |
| 0x38 | `REG_DESC_INFO` | R | `{8'h0, act_total_bytes[23:0]}` from last descriptor |
| 0x3C | `REG_Q8_DEBUG` | R | Q8 core debug word (see below) |
| 0x10 | `REG_Q8_NUM_GROUPS` | R/W | [3:0]: number of column groups (fallback, used when descriptor value = 0) |

**REG_DEBUG (0x28) bitfields:**
| Bits | Field | Description |
|------|-------|-------------|
| [31:28] | `state` | FSM state (see below) |
| [27] | `rd_done` | HP read master done (sticky, cleared on next LOAD_ACT entry) |
| [26] | `wr_done` | HP write master done (sticky, cleared on next WRITE_RES entry) |
| [25] | `rd_busy` | HP read master busy |
| [24] | `wr_busy` | HP write master busy |
| [23] | `q8_busy` | Q8 core busy |
| [22:20] | `wr_dbg_state` | Write master FSM state (3-bit) |
| [19:17] | `rd_dbg_state` | Read master FSM state (3-bit) |
| [16] | `q8_done` | Q8 core done |
| [15:8] | `wt_byte_idx[7:0]` | Weight byte index (lower 8 bits) — useful for tracking scale/act loading progress |
| [7:0] | `act_byte_idx[7:0]` | Activation byte index — useful for tracking act_buf fill level |

**REG_Q8_DEBUG (0x3C) bitfields:**
| Bits | Field | Description |
|------|-------|-------------|
| [31:28] | `state` | FSM state (same as REG_DEBUG[31:28]) |
| [27] | `q8_busy` | Q8 core busy |
| [26] | `q8_done` | Q8 core done (pulse) |
| [25] | `q8_start` | Q8 core start (pulse) |
| [24] | `q8_act_we` | Q8 activation write enable (active during COPY_ACT_TO_CORE) |
| [23:16] | `q8_res_idx` | Result read index (0..63) — active during READ_RES/READ_RES_ACC |
| [15:8] | (various) | `{copy_act_idx[2:0], q8_sc_we, sc_byte_idx[0], padding[3:0]}` — sc_byte_idx[0] toggles each scale byte pair |
| [7:0] | `wt_byte_idx[7:0]` | Weight byte index (same as REG_DEBUG[15:8]) |

### Descriptor Format (32 bytes, 8 words):

| Offset | Bits | Field | Description |
|--------|------|-------|-------------|
| 0 | [31:0] | `next_addr` | DDR address of next descriptor (0 = end of chain) |
| 4 | [31:0] | `weight_addr` | DDR address of Q8 weight data (4096 bytes + scales) |
| 8 | [31:0] | `act_addr` | DDR address of activation data (16-bit ints, tile rows × cols × 2) |
| 12 | [31:0] | `result_addr` | DDR address for result writeback (8 bytes/row) |
| 16 | [15:0] | `tensor_type` | 15 = CPU_OP (passthrough), 0 = Q8 compute, others reserved |
| 16 | [31:16] | (reserved) | Upper 16 bits reserved |
| 20 | [3:0] | `num_groups` | Q8 column groups (0 = use GP0 register fallback, 1 = single, 2+ = multi) |
| 20 | [31:4] | (reserved) | Upper 28 bits reserved |
| 24 | [23:0] | `act_total_bytes` | Total activation bytes to read from DDR |
| 28 | [31:0] | (reserved) | Reserved |

**FSM States (REG_DEBUG[31:28] or REG_Q8_DEBUG[31:28]):**
| Value | Name | Description |
|-------|------|-------------|
| 0 | `IDLE` | Waiting for REG_START write |
| 1 | `FETCH_DESC` | Starting descriptor read from DDR |
| 2 | `FETCH_DESC_W` | Waiting for descriptor read completion |
| 3 | `LOAD_ACT` | Starting activation data read from DDR |
| 4 | `LOAD_ACT_W` | Waiting for activation read completion |
| 5 | `WRITE_RES` | Starting result write to DDR |
| 6 | `WRITE_RES_W` | Waiting for result write completion |
| 7 | `DONE` | Chain complete (HEAD increments, next_addr check) |
| 8 | `LOAD_WEIGHT` | Starting Q8 weight data read from DDR |
| 9 | `LOAD_WEIGHT_W` | Waiting for weight read completion |
| 10 | `LOAD_SCALES` | Starting Q8 scale data read from DDR |
| 11 | `LOAD_SCALES_W` | Waiting for scale read completion, packing pairs into smem |
| 12 | `COPY_ACT_TO_CORE` | Copying act_buf to Q8 core activation registers |
| 13 | `COMPUTE` | Pulsing q8_start to begin computation |
| 14 | `COMPUTE_W` | Waiting for Q8 core done |
| 15 | `READ_RES` | Reading Q8 core results into act_buf (single-group) |
| 16 | `READ_RES_ACC` | Reading Q8 results, accumulating into acc_buf (multi-group) |
| 17 | `COPY_ACC_TO_BUF` | Copying acc_buf to act_buf for DDR writeback |

**Key experience: Debug register field usage patterns from hardware bringup:**
- `REG_Q8_DEBUG[26]` (q8_done) transitions 0→1 when computation completes; this is the most important signal for debug
- `REG_Q8_DEBUG[23:16]` (q8_res_idx) should count from 0→63 during result readback; if stuck at 0, the FSM never entered READ_RES/READ_RES_ACC
- `REG_Q8_DEBUG[15:8]` bit 0 (sc_byte_idx[0]) toggles each scale byte pair processed — if toggling stops, the scale loading DMA hung
- `REG_DEBUG[27]` (rd_done) and [26] (wr_done) are cumulative sticky bits — clear on next entry to LOAD_ACT/WRITE_RES respectively
- `REG_DEBUG[15:8]` vs [7:0] shows which FSM phase is active: [15:8] increments during LOAD_WEIGHT/LOAD_SCALES, [7:0] increments during LOAD_ACT

### Existing Verilog Cores:

| Core | Tile | Cycle/tile | Status |
|------|------|-----------|--------|
| `matmul_q8_core.v` | 64×896 | ~515 | ✅ Working |
| `matmul_q4k_core.v` | 56×256 | ~? | ✅ Working |
| `matmul_q5_0_core.v` | 4×896 | ~7170 | ✅ Working (2 cores, 4 rows/tile) |
| `matmul_int16_core.v` | 64×64 | 515 | ✅ Working |
| `matmul_top.v` | — | — | ✅ 5 cores instantiated (Q8, Q4K, Q5_0, Q6_K, INT16) |
| `hp_fsm_top.v` | HP FSM + Q8 + Q5_0 | N/A | ✅ Descriptor-chain DMA, Q8 compute 64×896 (14-group), Q5_0 4×896 (2-core) |

### Missing Verilog Cores:

| Core | Tile | Status |
|------|------|--------|
| None | — | All cores implemented ✅ |

## Current Status (2026-07-05) — 2-Core Q5_0, HW Verified

| Resource | Used | Available | % | Notes |
|----------|------|-----------|---|-------|
| Slice LUTs | **9,898** | 17,600 | **56.2** | +2,129 from Q5_0 (was 7,769 baseline) |
| LUT as Logic | **9,801** | 17,600 | **55.7** | |
| LUT as Memory | **97** | 6,000 | **1.62** | |
| Slice Regs | 12,781 | 35,200 | 36.3 | |
| BRAM18 | **33** | 120 | **27.5** | Q8(17) + 2×Q5_0(8+8) = 33 |
| DSP48E1 | **22** | 80 | **27.5** | Q8(16) + 2×Q5_0(3+3) = 22 |
| Slice | 4,387 | 4,400 | **99.7** | Tight — routing congestion |
| **WNS** | **-9.74 ns** | 10 ns | Violated | Works on HW despite violation |

**Bitstream sources:** `axihp_read_master.v` + `axihp_write_master.v` + `matmul_q8_core.v` + `matmul_q5_0_core.v` + `hp_fsm_top.v`

**Hardware tests:** All 13 tests PASS (9 baseline + 4 Q5_0). Q5_0 single desc (4 rows × 896) PASS. Q5_0 chain-of-2 (all-1s→all-0s) PASS. Mixed CPU_OP→Q5_0 chains PASS. Q8 tests unchanged.

### Hierarchical Resource Breakdown (Vivado routed)

```
Instance          Tot LUTs   %Total   FFs    BRAM18  DSP   Role
─────────────────────────────────────────────────────────────────────────
inst (FSM top)      4,464    47.0%   8,531     0       2   hp_fsm_top control logic
u_q8                2,148    22.6%   3,187    17      16   Q8 compute core
u_wr (write mstr)   1,141    12.0%    131      0       0   AXI HP write master
u_q5_core0            830     8.4%    167      8       2   Q5_0 core (rows 0-1)
u_q5_core1            766     7.7%    167      8       2   Q5_0 core (rows 2-3)
axi_lite (auto_pc)    392     4.1%    482      0       0   AXI protocol converter
u_rd (read mstr)      173     1.8%    116      0       0   AXI HP read master
─────────────────────────────────────────────────────────────────────────
Total               9,914   100%   12,781    33      22
```

**`inst (FSM top)` — 4,464 LUTs (47%)** is the binding constraint. Contains:
- Q8 weight/scale/act loading FSM with byte-unpack shift registers (~1,200 LUTs)
- Q5_0 weight loading (4,928-byte iteration, per-cycle bank/addr/we routing) (~800 LUTs)
- Q5_0 scale/act copy/unpack FSM (~300 LUTs)
- Buffer arrays: desc_buf(32B), act_buf(64×64b), acc_buf(64×48b) in FFs (~600 LUTs)
- AXI4-Lite slave + register file (~200 LUTs)
- Main FSM state decode + counters (~500 LUTs)
- Q8/Q5_0 dispatch branching (~400 LUTs)
- Q8 control signals (wmem/smem/acc BRAM steering) (~400 LUTs)

**Note on LUTRAM→BRAM conversion potential:** The 3 FF-based buffers (act_buf 4,096 b, acc_buf 3,072 b, desc_buf 256 b) total 7,680 bits — fitting in 1 RAMB18. Converting them would save ~7,400 FFs but **zero LUTs** (all 4,464 FSM LUTs are logic, not LUTRAM). Slice savings: ~50-70 slices (1-1.5%). Not the bottleneck.

**Why slices are full (99.7%) despite 56.2% LUT usage:**
- 473 CARRY4 chains lock LUT pairs into fixed slice positions
- 2,383 MUXF7 + 1,152 MUXF8 use dedicated mux per slice
- 307 unique control sets force slice-level partitioning
- LUTRAM in desc_buf/act_buf occupies SLICEM slices exclusively

### Q5_0 Pipeline Bugs Found and Fixed

1. **Off-by-one stale BRAM read** — Original LD→DEC set addresses via NBA at cycle end, BRAM read used OLD address. DEC always read data from previous iteration's block at block boundaries (ei%32==0). Fix: combinational BRAM addresses from `ei` (no rd_*_addr reg) + dedicated R cycle for 1-cycle read latency. Verified with non-uniform pattern test where q5_val = block % 16: row 0 was 5941 vs expected 5952 (loss of 11) before fix, 5952 after fix.

2. **MAC combinatorial timing** — Original `q5_mac()` had 3 DSP multiplies + shift + clamp + 16-bit multiply + 48-bit accumulate all in one cycle (31ns path). 4-stage pipeline splits it:
   - R: BRAM read latency
   - D1: `d_fp_w * q5_w` → `pipe_dq` (1 DSP)
   - D2: `pipe_dq * scale >> 8` + clamp → `pipe_dec` (1 DSP)
   - A: `acc[row] += pipe_dec * act_r` + ei++ (1 DSP + add)

### 2-Core Design (2026-07-05)

Reduced from 4 to 2 Q5_0 cores due to 99.84% slice utilization causing timing glitches on core 3. Each Q5_0 tile processes 4 rows (2 cores × 2 rows) instead of 8 rows. attn_q (896×896) requires 224 descriptors per layer instead of 112. All HW tests pass with no glitches.

| Metric | 4-Core (before) | 2-Core (after) | Delta |
|--------|----------------|---------------|-------|
| LUTs | 10,959 | 9,898 | -1,061 |
| BRAM18 | 49 | 33 | -16 |
| DSP | 28 | 22 | -6 |
| Slice | 4,393 | 4,387 | -6 |
| WNS | -7.78 ns | -9.74 ns | worse (routing var) |
| HW glitches | Row 7 = -232 | None | fixed |

## Two-Track Plan

### Track A: BRAM Conversion ✅ COMPLETE (2026-07-05)
All LUTRAM arrays in Q8 core converted to BRAM banks. Results:

| Memory | Before | After | BRAM18 Delta |
|--------|--------|-------|-------------|
| wmem | 1×512×64 BRAM | 8×512×8 BRAM | +7 (packed) |
| acc | 8× BRAM18 | 8× BRAM18 | 0 |
| smem | LUTRAM 128×16 | BRAM18 | +1 (packed) |
| act_reg | LUTRAM 64×16 | BRAM18 | 0 (packed) |
| **Total** | **9 BRAM** | **17 BRAM** | **+8** |

Savings: 3,512 LUTs (31%), 688 LUTRAM (92%), timing +0.314ns.

### Track B: Wider MAC (DSP Parallelism)

| Option | MACs | DSPs | LUTs | Cycles/tile |
|--------|------|------|------|-------------|
| Current | 8 | 16 | 7,769 | 524 |
| 16× col | 16 | 32 | ~10,500 | 260 |
| 32× col | 32 | 64 | ~14,000 | 135 |

### Roadmap
1. ✅ Q8 BRAM-acc baseline: all core unit tests PASS (Q8 6/6, Q4K 4/4, Q5_0 32/32, Q6_K 97/97, HP FSM 7/7)
2. ✅ Track A: smem→BRAM, act_reg→BRAM, wmem→8×BRAM (6-stage pipeline proven)
3. ✅ Rebuild bitstream, verify all 9 HW tests — ALL PASS
4. ✅ Measured: 3,512 LUTs saved (31%), 688 LUTRAM (92%), WNS 0.601ns
5. ✅ Q5_0 integration: 4× Q5_0 cores in hp_fsm_top, descriptor tensor_type=1 dispatch, 4-stage pipeline, HW test PASS (2026-07-05)
6. ▶ Multi-core integration (Q4K/Q6_K/INT16) with shared BRAM weight loading

## Build & Run Commands

```bash
# PATH setup
$env:Path = "D:\Program Files\Git\bin;$env:Path"   # git

# Verilog tests (iVerilog — d:\iVerilog\bin)
make -C verilog all                     # Q8 + Q4K + Q5_0 + INT16 smoke

# Vivado xsim (C:\Xilinx\Vivado\2023.1\bin\xsim.bat)
xsim tb_hw_fsm --runall                  # HP FSM descriptor-chain test

# Vivado batch build
C:\Xilinx\Vivado\2023.1\bin\vivado.bat -mode batch -source vivado_integration/build_bd.tcl

# JTAG load via XSDB (single session: program + init + run + capture)
C:\Xilinx\Vivado\2023.1\bin\xsdb.bat vivado_integration\sw\run_hp_fsm_comprehensive.tcl

# NOTE: Power-cycle the board before running ps7_init via XSDB!
# ps7_pll_init hangs if PLLs are already configured from a prior session.

# JTAG load via Vivado HW Manager (clears DAP sticky errors)
Copy system_wrapper.bit from proj_bd/ to current dir, then run Vivado HW manager

# C++ integration test
g++ -std=c++14 -O2 -I sim -I gguf -I . sim/test_integration.cpp -lpthread -o /tmp/ti
/tmp/ti

# Build inference engine
g++ -std=c++17 -pthread -O2 -I sim -I gguf -I . sim/tmac_gguf.cpp sim/matmul_q8.cpp -o sim/tmac_gguf

# FPGA simulation flags (individual paths can be combined)
echo "9707" | ./sim/tmac_gguf models/model.tmac --fpga-q8              # Q8_0 only
echo "9707" | ./sim/tmac_gguf models/model.tmac --fpga-q5-0           # Q5_0 only
echo "9707" | ./sim/tmac_gguf models/model.tmac --fpga-q6-k          # Q6_K only
echo "9707" | ./sim/tmac_gguf models/model.tmac --fpga-q4k          # Q4_K only
echo "9707" | ./sim/tmac_gguf models/model.tmac --fpga-q5-0 --fpga-q6-k --fpga-q4k  # combined
echo "9707" | ./sim/tmac_gguf models/model.tmac --fpga-q8 --fpga-q5-0 --fpga-q6-k --fpga-q4k  # all paths

# Full test suite
bash scripts/test_integration.sh

# Model conversion
python3 scripts/extract_tmac.py models/qwen2-0_5b-instruct-q4_k_m.gguf /tmp/model.tmac
```

## File Inventory

### Verilog RTL
- `verilog/matmul_top.v` — Quad-core top: AXI4-Lite slave (inline, replaces orphan axilite_slave.v), 8192-byte weight_buf, loading FSM, mode mux (Q8/Q4K/Q5_0/Q6_K/INT16)
- `verilog/matmul_q8_core.v` — Q8_0 compute core: 8×512×8 wmem (BRAM banks), dequant LUT, 6-stage pipeline
- `verilog/matmul_q4k_core.v` — Q4_K block decode: 2304-byte block buffer, S24.8 fixed-point, 56×256 tile
- `verilog/matmul_q5_0_core.v` — Q5_0 block decode: 8×896 tile, 4× parallel cores, 4-stage pipeline (R→D1→D2→A), 1 DSP multiply/stage, combinational BRAM addresses (off-by-one fix)
- `verilog/matmul_q6_k_core.v` — Q6_K block decode: 32×256 tile, 32 blocks/tile, super_scale + per-sub-block scales
- `verilog/matmul_int16_core.v` — General INT16×INT16 core: 512×128-bit wmem, 3-stage FSM
- `verilog/dequant_lut.v` — Q8_0 dequant ROM (standalone, not instantiated)
- `verilog/systolic_8x8.v` — 8×8 systolic array (standalone, not used)

### Verilog Testbenches
- `verilog/tb_matmul_q8.v` — Q8 core tests
- `verilog/tb_matmul_q4k.v` — Q4K core tests
- `verilog/tb_minimal_q4k.v` — Q4K smoke test
- `verilog/tb_int16_smoke.v` — INT16 smoke test
- `verilog/tb_cosim.v` — Q8_0 co-simulation
- `verilog/tb_cosim_q4k.v` — Q4_K co-simulation
- `verilog/tb_matmul_q5_0.v` — Q5_0 core tests (fabricated patterns)
- `verilog/tb_matmul_q6_k.v` — Q6_K core tests (fabricated patterns)
- `verilog/tb_cosim_q5_0.v` — Q5_0 co-simulation (waits for tile dump)
- `verilog/tb_cosim_q6_k.v` — Q6_K co-simulation (waits for tile dump)
- `verilog/tb_hw_fsm_comprehensive.v` — HP FSM all 7 tests (HEAD-based wait_done)
- `verilog/tb_hp_fsm_q5_0.v` — HP FSM Q5_0 dispatch test (all-1s pattern, chain of 2, mixed CPU_OP+Q5_0, chain of 3)
- `verilog/tb_q5_off_by_one.v` — Q5_0 off-by-one BRAM bug verification (non-uniform patterns)
- `verilog/test_hp_loopback.v` — 32-bit HP loopback testbench (ARSIZE=2 proven)
- `verilog/sim_ddr_axi_hp.v` — AXI HP DDR model for simulation

### C++ Simulation
- `sim/tmac_gguf.cpp` — Full inference pipeline, dispatch by tensor type
- `sim/fpga_sim.hpp` — `MatmulAccel`, `axi_vecmul_tile_*` functions, decode logic
- `sim/matmul_q8.cpp` — Q8_0 logits path wrapper
- `sim/test_integration.cpp` — Integration tests
- `sim/chat.py` — Chat interface

### Scripts
- `scripts/extract_tmac.py` — GGUF→TMAC converter
- `scripts/ground_truth_v2.py` — Ground truth generation
- `scripts/verify_layers_fast.py` — Layer verification
- `scripts/design_iteration.sh` — Vivado iteration loop
- `scripts/feedback_parser.py` — Report parser
- `scripts/test_integration.sh` — Test suite

### Documentation
- `verilog/DESIGN.md` — Architecture, timing
- `docs/architecture.md` — Model, quantization formats
- `docs/FPGA_PERFORMANCE_ANALYSIS.md` — Performance analysis
- `docs/Q4_K_IMPLEMENTATION_PLAN.md` — Original plan (outdated)

## Target Board: MicroPhase Z7-Lite

- **Part**: `xc7z010clg400-1` (Zynq-7010, 28nm, 432 CLBs, 240 DSP48E1, 4.9 Mb BRAM)
- **DDR3**: 512 MB, Micron **MT41J256M16 RE-125** (4 Gbit, 16-bit bus, 15 row × 10 col × 3 bank)
- **DDR speed**: DDR3-1066F (533 MHz core, CL=7, CWL=6, tRCD=7, tRP=7, tRAS=35ns, tRC=48.91ns, tFAW=40ns)
- **PS7 clock config**: 33.333 MHz crystal → ARM=666.667 MHz, DDR=533.333 MHz, PL=100 MHz
- **UART console**: UART0 (MIO 14/15) at 115200 baud
- **Debug**: JTAG via Digilent HS-2 on the onboard FTDI/JTAG bridge
- **Peripherals**: No Ethernet, no SD card (bare-metal only)

**DDR address map** (from PS7 config): 0x0010_0000 to 0x2000_0000 (512 MB), CPU accesses at 0x0000_0000 alias after MMU/cache setup; bare-metal uses physical 0x0010_0000 base.

## Phase 1: AXI4-Lite + HP (High Performance Port)

### Status
- **AXI4-Lite (GP0) control path** ✅ Verified on hardware — all register R/W correct
- **HP0 write path** ✅ Verified on hardware — STATUS=0x1E, DEBUG=0x0F, correct data at DDR target
- **HP0 read path** ✅ Verified on hardware — 16-beat burst (ARSIZE=2) returns correct data
- **HP FSM descriptor-chain** ✅ **PASSED (2026-06-25)** — single descriptor: descriptor fetch → act load → result writeback, 8 words match, 19ms, ARSIZE=2/AWSIZE=2
- **HP FSM comprehensive (7 tests)** ✅ **ALL PASSED ON HARDWARE (2026-06-27)** — basic 64B, min 8B, 128B 2-burst, 256B 4-burst, chain of 2 desc, chain of 3 desc, re-start from DONE. All STATUS=0x300, all patterns verified.
- **Testbench fix: wait_done polls HEAD instead of STATUS bits** — cumulative STATUS bits (rd_done/wr_done) stay set across descriptors, causing premature exit in chain tests. Fixed by polling HEAD register.
- **ACP** 🔄 Not needed — HP works reliably when PS7 is freshly initialized
- **Phase 1 complete — HP descriptor-chain DMA proven on hardware** across all edge cases (min/max sizes, chains, restart). Ready for Phase 2: Q8 compute integration.
- **Phase 2 (Q8 compute)** ✅ **ALL 9 TESTS PASS ON HARDWARE** — Q8 pipeline timing fix (WNS +0.550), sc_byte_idx reset bug fixed, all-1s pattern. Three bugs fixed (q8_wt_din reg, col_group init, act_remaining). Test 9a multi-group 2-group 64×128 tile PASS (all 64 rows = 128). Multi-group verified on HW (2026-07-02). **rd_ready handshake fix (2026-07-04):** Changed LOAD_WEIGHT_W from `rd_ready <= 1` to `rd_ready <= rd_valid` — the continuous-high read-enable caused the read master's `rvalid` to self-clear in 0 cycles (NBA race). All 9 HW tests pass after 64-bit word write change. Phase 2 complete.

### Critical: ps7_init re-execution hang
`ps7_pll_init_data_3_0` **hangs if PLLs are already configured** from a prior session. The PLL reset sequence (bypass→power-down→reset→wait-for-lock) can't re-lock when the PLLs are already locked from a previous session. This leaves the PS7 in a partially-configured state and all subsequent ps7_init attempts also hang (DDR init's `mask_poll 0xF8000B74 0x00002000` waits for calibration that depends on PLL clock).

**Workaround:** Always power-cycle the board before running ps7_init via XSDB. A processor-only reset (`rst -processor`) is insufficient — the PLLs must be in reset-init state for ps7_pll_init to succeed.

**Key observation:** HP0 register reads return 0x00000000 after a failed ps7_init attempt, even though the OCM code wrote valid non-zero values. This is because the PS7 AHB interconnect enters an inconsistent state when PLLs are partially configured, and DAP read transactions return 0.

### Batch Build Framework

| Script | Purpose | Command |
|--------|---------|---------|
| `vivado_integration/build_bd.tcl` | Vivado batch build | `C:\Xilinx\Vivado\2023.1\bin\vivado.bat -mode batch -source vivado_integration/build_bd.tcl` |
| `vivado_integration/sw/rebuild.tcl` | XSCT: rebuild Vitis app | `C:\Xilinx\Vitis\2023.1\bin\xsct.bat vivado_integration/sw/rebuild.tcl` |
| `vivado_integration/sw/run_hp_fsm_comprehensive.tcl` | XSDB: all 7 HP FSM tests (basic, min, 2-burst, 4-burst, chain 2, chain 3, restart) + Test 8-9 (Q8 all-1s, Q8 multi-group) | `C:\Xilinx\Vivado\2023.1\bin\xsdb.bat vivado_integration/sw/run_hp_fsm_comprehensive.tcl` |
| `vivado_integration/sw/run_hp_fsm_q5_0.tcl` | XSDB: Q5_0 compute test (all-1s, 8 rows, 896 each) | `C:\Xilinx\Vivado\2023.1\bin\xsdb.bat vivado_integration/sw/run_hp_fsm_q5_0.tcl` |

### Debug Workflow

```
C:\Xilinx\Vivado\2023.1\bin\vivado.bat -mode batch -source vivado_integration\build_bd.tcl    # Step 1: build bitstream
C:\Xilinx\Vivado\2023.1\bin\xsdb.bat vivado_integration\sw\run_hp_fsm_comprehensive.tcl      # Step 2: FPGA + ps7_init + test
```

### PS7 Config Changes

`ps7_post_config_3_0` in `ps7_init.tcl` was modified to enable AFI1:
```tcl
mwr -force 0xF8009000 0x00000003  ;# AFI1 enable + bypass FIFO
mwr -force 0xF8009008 0x00000001  ;# write channel enable
mwr -force 0xF800900C 0x00000001  ;# read channel enable
```

### HP Loopback Test — ARSIZE=3 Rejected, ARSIZE=2 Proven (2026-06-20)

**Initial hypothesis (WRONG):** ARSIZE=2/AWSIZE=2 (4-byte narrow transfers) on Zynq 64-bit HP interconnect was thought to cause byte lane remapping issues. Tried ARSIZE=3/AWSIZE=3.

**Board test with ARSIZE=3: RDATA[63:32]=0** — Zynq-7010 with x16 DDR3 caps HP0 at 32-bit. Upper 32 bits of each 8-byte beat are always 0, losing every other 32-bit word. ARSIZE=3 REJECTED.

| Fix Attempt | Result | Detail |
|-----|--------|--------|
| done bits sticky | ✅ | `reg_status[15:8]` no longer cleared in DONE state |
| ARSIZE=3 (read master) | ❌ REJECTED | RDATA[63:32]=0 on hardware, data corruption |
| AWSIZE=3 (write master) | ❌ REJECTED | Same upper-half issue on write side |
| Bitstream rebuilt | ✅ | 0 errors, synth 50s + impl 2:31 |

**Actual fix:** Reverted to ARSIZE=2/AWSIZE=2, accepting the 32-bit nature of HP0 on this board.

### 32-bit HP mode (2026-06-20) — THE WORKING SOLUTION

**Finding: Zynq-7010 HP0 is 32-bit only with x16 DDR3.** Despite `PCW_S_AXI_HP0_DATA_WIDTH=64` set before `apply_bd_automation`, RDATA[63:32] is always 0 on hardware. The PS7 silicon ignores the 64-bit width parameter when the DDR bus is x16-wide. `AFI0_CTRL[7:6]` (64-bit enable) is also read-only — confirmed by write-verify loop.

**Design change:** Switched to full 32-bit AXI mode:

| Component | Change |
|-----------|--------|
| `axihp_read_master.v` | ARSIZE=2 (4 bytes/beat), always captures `m_axi_rdata[31:0]` (no beat[0] alternation since HP0 doesn't do narrow-transfer byte lane remapping — RDATA[63:32] is always 0) |
| `axihp_write_master.v` | AWSIZE=2 (4 bytes/beat), splits each 64-bit top word into two single-beat AXI transactions. wready asserted **once per word** (in W_L only, not W_U). W_U proceeds without wready handshake using `hold_wdata[63:32]`. 5-state FSM: IDLE→AW_L→W_L→B_L→AW_U→B_U. |
| `hp_loopback_top.v` | Word_idx advances once per 64-bit word (single wready assertion). |
| `run_hp_loopback.tcl` | REG_RD_LEN=15 (16 beats × 4 bytes = 64 bytes). AFI0_CTRL=0x01 (bits[7:6] omitted). |

**Simulation:** 32-bit HP loopback **passes** in iVerilog — reads 16 bytes (pattern 0x00..0x0F) into word_buf, writes 8 × 64-bit words to DDR offset 0x40, verifies W0_lo/W0_hi/W1_lo/W1_hi and DDR[8]/DDR[9] match expected values.

| Section | Status |
|---------|--------|
| Read path (16 beats × 4 bytes) | ✅ Verified — byte stream correct, word assembly correct |
| Write path (8 words, 16 × 32-bit AXI transactions) | ✅ Verified — lower/upper half split correct, DDR stored correctly |
| Buffer dump registers | ✅ Verified — all 16 × 32-bit debug regs match expected |
| DDR readback after write | ✅ Verified — all 8 words match source pattern |

**Bug fixed:** Write master previously asserted wready twice per word (in both W_L and W_U), causing `word_idx` to advance by 2 per word, skipping every other word. Fixed by removing wready assertion in AW_U/W_U — W_U immediately sends `hold_wdata[63:32]` without handshake.

**Board test (2026-06-20):** HP loopback **PASSES** on hardware with Zynq-7010 x16 DDR3:
| Test | Result |
|------|--------|
| Read path (16 beats × 4 bytes ARSIZE=2) | ✅ All 16 debug registers match expected |
| Write path (8 words, 16 × 32-bit AXI transactions) | ✅ All 8 words at DDR destination match |
| PATTERN_OFF 2-beat read | ✅ DBG_LO=0xA5A5A5A5 DBG_HI=0x5A5A5A5A |
| Timing | 9ms loopback (incl. read + write) |

**Key insight for WSTRB:** On the 64-bit HP port with AWSIZE=2 (32-bit narrow writes), address bit A[2] selects byte lanes: A[2]=0 → WDATA[31:0] with WSTRB[3:0], A[2]=1 → WDATA[63:32] with WSTRB[7:4]. The original design sent both halves on WDATA[31:0] with WSTRB[3:0], corrupting the upper half write.

### Latest Debug Session (2026-06-25)

**HP FSM descriptor-chain test PASSES on hardware.** After reverting read/write masters to ARSIZE=2/AWSIZE=2 (proven working by loopback test) and rebuilding the Vivado bitstream:

| Test | Status | Detail |
|------|--------|--------|
| ps7_init (fresh power-cycle) | ✅ | PLL_STATUS=0x3F (all locked), DDR calibration OK |
| PL clock verified | ✅ | Clock counter = 0x019D0C77 (~27M cycles) |
| AFI config (DAP writes) | ✅ | CTRL=0x05, STATUS=0x0F00 |
| GP0 access | ✅ | All register readbacks correct |
| Descriptor fetch (HP read, 8 beats) | ✅ | All 8 descriptor words parsed correctly |
| Act load (HP read, 16 beats, 64 bytes) | ✅ | Correct data from 0x00101000 |
| Result writeback (HP write, 8 words) | ✅ | All 8 result words match expected patterns |
| Chain completion | ✅ | STATUS=0x300 (rd_done=1, wr_done=1), DEBUG=0x70000F40 (state=7/DONE) |
| Timing | 19ms | Config overhead, negligible for production |

**Key findings:**
- ARSIZE=2/AWSIZE=2 proven end-to-end: descriptor fetch → act load → result writeback
- HP FSM correctly sequences IDLE→FETCH_DESC→LOAD_ACT→WRITE_RES→DONE
- GP0 write to REG_START auto-cleared by FSM (expected — FSM leaves IDLE and clears start bit)
- Previous "REG_ACT_INFO=0" failure was caused by ARSIZE=3 read master misaligning descriptor data — 8-byte beat consumption with only 4 valid bytes/beat caused act_addr parsed from wrong position

**Implications:**
- Phase 1 (AXI4-Lite + HP port) fully verified on hardware
- HP FSM is ready as building block for weight loading in compute pipeline
- Next: integrate HP read master with matmul_top's weight_buf loading, or proceed to single-layer compute

## Phase 2: Q8 Compute on Hardware

### Q8 Pipeline Fix (2026-06-28)

**Problem:** The Q8 core's dequant (24-bit multiply) and p2_partial (16×16 multiply) were in a single combinatorial path, causing a critical path that violated 100 MHz timing (WNS ~ -0.7 ns).

**Fix:** Split into two pipeline stages by adding intermediate `dq_deq[0:7]`/`dq_act`/`dq_row_base`/`dq_valid` registers between the dequant multiply and the p2_partial multiply. Also added `DRAIN3` state for the extra pipeline depth.

**Result:** WNS improved to +0.550 ns — timing closure achieved with 0 errors.

### sc_byte_idx Reset Bug (2026-06-28)

**Problem:** The `sc_byte_idx` counter was being reset to 0 at the start of every scale-loading burst (`sc_byte_idx <= 0` at entry to LOAD_SCALES). This caused each of the 4 bursts to overwrite `smem[0..31]` rather than writing to `smem[0..127]` progressively. Only the first 32 of 128 smem entries contained valid scale data.

**Fix:** Removed the `sc_byte_idx <= 0` assignment between bursts (only keep it between column groups in READ_RES_ACC). The counter now persists across all 4 bursts within a group, correctly writing all 128 smem entries.

**Board result:** Tests 1-7 (baseline DMA) all PASS unchanged. Test 8 (Q8 all-1s) now produces **all 64 rows = 64** — previously rows 0-15 gave 32 (wrong) and rows 16-63 gave 0.

### Multi-group Bug Fixes (2026-07-01)

Three bugs in the multi-group Q8 iteration were found and fixed during iVerilog simulation of the comprehensive test suite:

1. **`q8_wt_din` not registered** — The read master's `rd_data` is a combinatorial wire, while `q8_wt_we` and `q8_wt_addr` are registered via NBA assignments. Without registering `q8_wt_din`, the data changed combinatorially one cycle before we/addr settled, writing byteₙ+₁ to addressₙ (systematic byte-offset-1 error). Fix: added `reg [7:0] q8_wt_din` and assigned `q8_wt_din <= rd_data` alongside `q8_wt_we <= rd_valid` at line 450 of `hp_fsm_top.v`.

2. **`col_group <= 0` in COMPUTE_W** — The group counter was unconditionally reset to 0 after every COMPUTE_W completion, preventing multi-group iteration from ever reaching `q8_num_groups - 1`. Fix: removed `col_group <= 0` from the COMPUTE_W → READ_RES_ACC transition.

3. **`act_remaining <= 128` in READ_RES_ACC** — A hardcoded 128-byte constant was left from a single-group prototype. When re-entering the multi-group loop for groups > 0, this caused activation loading to use the wrong size. Fix: changed to `act_remaining <= act_total_bytes`.

**Other fixes:**
- `tb_hw_fsm_comprehensive.v`: descriptor `tensor_type` was 0 (default), causing all 7 tests to enter LOAD_WEIGHT path and read X from address 0 instead of CPU_OP → LOAD_ACT. Fixed to `32'h0000000F` (tensor_type=15).
- `tb_hw_fsm_comprehensive.v`: added `$dumpfile`/`$dumpvars` for VCD waveform generation.
- `matmul_q8_core.v`: added `__ICARUS__` simulation-only `initial` block to clear `wmem[0:511]` and `acc[0:63]`, preventing X on `res_dout`.
- `Makefile`: added `matmul_q8_core.v` to `HPFSM_SRC` (was missing since Q8 integration into `hp_fsm_top`).

**Verification:** All 7 comprehensive tests pass with correct data; Q8 core 6/6, Q4K 4/4, Q5_0 all rows, Q6_K 97/97 tests pass. INT16 smoke failure is pre-existing and unrelated.

### Multi-group 64×896 Tile (2026-06-28)

**Problem:** The original Q8 compute path only handled a single column group (64 columns). The full tile requires 14 groups × 64 cols = 896 columns.

**Implementation in `hp_fsm_top.v`:**
- Added `reg_q8_num_groups[3:0]` (R/W at address 0x40) — number of column groups
- Added `acc_buf[0:63]` (64 × 48-bit signed) — running accumulator across groups
- Added `col_group[3:0]` counter — tracks which group is being processed
- Added `READ_RES_ACC(16)` state — first group stores directly, subsequent groups add
- Added `COPY_ACC_TO_BUF(17)` state — copies acc_buf to act_buf for DDR writeback after final group
- Scale/activation address calculations now include `col_group × 128/256` offset
- FSM loops: LOAD_SCALES → LOAD_ACT → COPY_ACT_TO_CORE → COMPUTE → READ_RES_ACC (per group), then COPY_ACC_TO_BUF → WRITE_RES

**Status:** RTL complete, syntax-verified with iVerilog (0 errors). **Three bugs fixed (2026-07-01):** q8_wt_din unregistered (NBA timing), col_group reset in COMPUTE_W, act_remaining hardcoded to 128. Pending: build bitstream and hardware test.

### Previous Debug Session (2026-06-19)

**Power-cycle unblocks HP write.** After all ps7_init attempts hung (PLL re-lock failure on pre-configured PLLs), power-cycling the board allowed a clean ps7_init to complete. The HP write path verified functional:

| Test | Status | Detail |
|------|--------|--------|
| Load bitstream | ✅ | `fpga -file` completes |
| Full ps7_init | ✅ | All 6 functions, PLL lock ~ms |
| OCM AFI config | ✅ | Marker at 256ms, AFI0_CTRL=0x05, PART=0x44 |
| HP write (16 beats) | ✅ | STATUS=0x1E, DEBUG=0x0F, DDR[0]=0xA5A5A5A5 |

**Critical findings:**
- `ps7_pll_init_data_3_0` **hangs when PLLs are already configured** — the reset+re-lock sequence fails because PLLs are already locked, leaving the system in a partial state
- `ps7_ddr_init_data_3_0` then hangs at `mask_poll 0xF8000B74 0x00002000` (DDR calibration status) because DDR PLL clock is not recovered
- After a hang, ALL subsequent ps7_init attempts also fail — PS7 has no clean recovery path without power-cycle
- When DAP reads return 0x00000000 for all PS7 registers (including AFI), it signals the AHB interconnect is broken from a partial PLL init
- `HP_CLK_CTRL` (0xF800016C)=0x00000000 yet HP write works — HP is clocked from DDR clock domain, not a separate gate

**Verilog/design implications:**
- AFI0_FIFO_PARTITION is freely writable via OCM code (0x44 = 4R+4W) — NOT locked by boot ROM
- The earlier "HP dead" diagnosis was a false positive caused by reading registers after failed ps7_init
- ACP switch is unnecessary; HP0 will be used for all PL↔DDR traffic

### Relevant Files

- `vivado_integration/build_bd.tcl`: Vivado batch build — HP0 config (`PCW_S_AXI_HP0_DATA_WIDTH=64`) set before `apply_bd_automation`. Sources `hp_fsm_top.v` + `axihp_read_master.v` + `axihp_write_master.v`.
- `vivado_integration/rtl/hp_fsm_top.v`: HP descriptor-chain FSM — AXI4-Lite slave, desc_buf (32B), act_buf (512B), Q8 matmul core + 64×896 tile (14 groups), multi-group accumulator (64×48-bit acc_buf), 32-bit HP read/write masters. 18-state FSM: IDLE→FETCH_DESC→LOAD_WEIGHT→LOAD_SCALES→LOAD_ACT→COPY_ACT_TO_CORE→COMPUTE→READ_RES_ACC (×groups)→COPY_ACC_TO_BUF→WRITE_RES→DONE.
- `vivado_integration/sw/run_hp_fsm_comprehensive.tcl`: XSDB flow — all 7 HP FSM tests (basic, min 8B, 128B 2-burst, 256B 4-burst, chain of 2, chain of 3, re-start). Polls HEAD register for completion.
- `vivado_integration/sw/run_hp_fsm_q5_0.tcl`: XSDB flow — Q5_0 all-1s test. Loads weight/scales/acts, sets tensor_type=1 descriptor, verifies 8 rows = 896.
- `vivado_integration/sw/regs.h`: Register map
- `vivado_integration/ps7_init.tcl`: Modified — AFI1 + LVL_SHFTR_EN config in ps7_post_config
- `verilog/axihp_read_master.v`: HP read master — ARSIZE=2 (4 bytes/beat), always captures RDATA[31:0], byte-stream output. DRAIN state per-beat. 32-bit mode.
- `verilog/axihp_write_master.v`: HP write master — AWSIZE=2, splits 64-bit word into two 32-bit single-beat AXI writes. wready once per word. 5-state FSM.
- `verilog/matmul_int16_core.v`: INT16 compute core (verified standalone)
- `verilog/test_hp_loopback.v`: 32-bit mode simulation testbench — DDR model with 32-bit read/write granularity. Passes full loopback.
- `vivado_integration/proj_bd/matmul_bd.runs/impl_1/system_wrapper.bit`: Synthesized bitstream (reordered PS7 config)
- `D:/Users/u/workspace/tmac/Debug/tmac.elf`: Vitis ELF loaded by XSDB
- `docs/debug_log.md`: Full debug history
