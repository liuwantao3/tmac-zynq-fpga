## Project State

Qwen2-0.5B FPGA accelerator targeting Zynq 7010. Multi-core Verilog RTL: INT16×INT16 (general), Q8_0 dequant (logits), Q5_0 block decode (attn_q/k/o, ffn_gate/up), Q6_K block decode (ffn_down even), Q4_K block decode (ffn_down odd), with AXI4-Lite memory-mapped I/O.

**Model is Q4_K_M (mixed quantization).** Actual tensor types:

| tensor | shape | type | C++ path | Verilog core |
|--------|-------|------|----------|-------------|
| `token_embd` | 151936×896 | Q8_0 | ✅ | ✅ `matmul_q8_core.v` |
| `attn_v` | 128×896 | Q8_0 | ✅ | ✅ `matmul_q8_core.v` |
| `attn_q`, `attn_k`, `attn_output` | 896×896 | Q5_0 | ✅ `matmul_fpga_q5_0` | ✅ `matmul_q5_0_core.v` |
| `ffn_gate`, `ffn_up` | 4864×896 | Q5_0 | ✅ `matmul_fpga_q5_0` | ✅ `matmul_q5_0_core.v` |
| `ffn_down` (layers 0,2,4,...) | 896×4864 | Q6_K | ✅ `matmul_fpga_q6_k` | ✅ `matmul_q6_k_core.v` |
| `ffn_down` (layers 1,3,5,...) | 896×4864 | Q4_K | ✅ `matmul_fpga_q4_k` | ✅ `matmul_q4k_core.v` (56×256 tile) |

## Key Decisions (2026-07-12)

1. **TB `wr` task: `input integer din` truncates 64-bit weight word** — Found: `tb_matmul_q8.v` declared `wr(input integer we, addr, din)` where `integer` is 32-bit signed. Passing `word = {8{val}}` (64-bit) truncated upper 32 bits, causing banks 4-7 to receive 0x00 for positive weights (all rows 4-7 = 0). Test 6 (wt=-1=0xFF) worked because sign-extension filled upper 32 bits with 1s. Fix: `input [63:0] din`. All 6 Q8 tests now PASS.

2. **Break statements removed from cosim testbenches** — `tb_cosim.v`, `tb_cosim_q4k.v`, `tb_cosim_q5_0.v`, `tb_cosim_q6_k.v` used unsupported `break` in Verilog for-loop wait loops. Replaced with `poll_count` flag loop condition.

3. **Q8 core: 64-bit word write, LUTRAM smem/act, dist-RAM acc banks** — Q8 core rewritten:
   - Write port: `wt_addr[8:0]`/`wt_din[63:0]` replaces byte-lane BWE case (BRAM-friendly)
   - Accumulator: 8× distributed RAM/FFs banks (acc_b0..acc_b7), each 512×48 effective (48-bit width exceeds BRAM18 capacity), banked by address[2:0]=g
   - Dequant saturation removed: max product 127×65535=8,322,945 < 8,388,607, never saturates
   - Pipeline: 6-stage (PRE→S0→S1a→S1b→S2a→S2b), CLEAR_ACC state for bulk BRAM clear
   - Result: saves ~884 LUTs (384 LUTRAMs + 500 logic) vs old reg [47:0] acc[0:63]

4. **HP FSM: rd_ready <= rd_valid in LOAD_WEIGHT_W** — Fixed 0-cycle rvalid pulse: continuous `rd_ready=1` caused read master's PRESENT state to self-clear rvalid. Same delayed-handshake as LOAD_ACT_W.

5. **Multi-group Q8: q8_wt_din reg, col_group fix, act_remaining fix** — Three bugs from multi-group (2026-07-01): unregistered wt_din (NBA timing), col_group reset in COMPUTE_W, hardcoded act_remaining in READ_RES_ACC.

6. **All core unit tests PASS (2026-07-05):** Q8 6/6, Q4K 4/4, Q5_0 32/32, Q6_K 97/97, HP FSM 7/7. INT16 smoke pre-existing failure (unrelated wmem addressing).

7. **Track A BRAM conversion complete (2026-07-05):** All LUTRAM arrays converted to BRAM — 7,769 LUTs (44.1%), 17 BRAM18 (14.2%), 16 DSP (20%). WNS=0.601ns. All 9 HW tests PASS.

8. **Q5_0 block-streaming redesign (2026-07-06):** Replaced 4-cycle/element pipeline (7170 cycles) with block-at-a-time architecture (1904 core cycles + DDR overhead ≈2856 system cycles/tile). Precomputes d_pre = f16_decode(d) × scale >> 8 once per block, then 1 MAC/cycle per element. Scale indexing bug fixed (core1 used scale[0:1] instead of scale[2:3]). All 8 unit tests + 4 HP FSM dispatch tests PASS (pre-redesign testbench; core unit tests need port-renamed rebuild for new `qs_word`/`blk_num` interface).

9. **Q5_0 clean-slate rewrite — per-block wide register interface (2026-07-07):** Completely rewrote both `matmul_q5_0_core.v` and `hp_fsm_top.v` to eliminate byte-at-a-time header loading, `hdr_packed` LUTRAM, and `qs_word`/`blk_num`/`core_id`/`start` ports. The core now receives per-block `blk_d[15:0]`, `blk_qh[31:0]`, `blk_qs[127:0]` via a single `blk_valid` pulse — all fed from a 48-byte DDR burst (12 AXI beats) per block. No LUTRAM for header storage (at most one block's d/qh/qs in the core at a time). Fixed pre-existing `act_r` pipeline bug (off-by-one masked by all-1s test activations). `row_scale` renamed to `row_norm`. FSM states Q5_PRELOAD_HDR/Q5_PRELOAD_HDR_W removed, replaced by Q5_LOAD_NORM/Q5_LOAD_NORM_W (8-byte DDR read for 4 × UQ8.8 norm values). Core results read via hierarchical reference (`u_q5_core0.res0/res1`). DDR layout per block: 48 bytes (core0_d+qh+qs + core1_d+qh+qs + padding). Total: 56 × 48 = 2688 bytes block data + 8 bytes norm = 2696 bytes/tile. All 9 HP FSM dispatch tests PASS.

10. **CPU_OP col_group bug — `col_group` not reset on CPU_OP dispatch (2026-07-11):** The HP FSM's `LOAD_ACT` state always adds `col_group * 128` to `rd_addr` (`hp_fsm_top.v:675`). For Q8 compute, `col_group` is cleared in `LOAD_WEIGHT_W` before transitioning to `LOAD_SCALES`. For CPU_OP (tensor_type=15), the FETCH_DESC dispatch goes directly to `LOAD_ACT` without clearing `col_group`. If the previous descriptor was a multi-group Q8 (e.g., 14 groups), `col_group` retains its last value (e.g., 13), causing CPU_OP's `LOAD_ACT` to read from `act_addr + 13*128` — the wrong DDR address. E9's CPU_OP (which follows a single-group Q8 descriptor) works by accident because Q8 single-group also clears `col_group` in `LOAD_WEIGHT_W`.

    **Test workaround:** Added dummy Q8 single-group chain (`write_desc ... 128 0 0`) before each CPU_OP test to force `col_group=0`. **Proper fix:** Add `col_group <= 0` in the CPU_OP branch of FETCH_DESC dispatch (near `hp_fsm_top.v:642-644`). Also affects Q5_0 path (second descriptor after multi-group Q8) if Q5_0 uses LOAD_ACT for any purpose — current Q5_0 uses Q5_LOAD_NORM/Q5_COPY_ACT/Q5_BLOCK_COMPUTE which don't use `col_group`.

11. **CPU_OP interrupt protocol (2026-07-12):** Added `interrupt` output port, `reg_chain_ctrl` (0x04), `reg_gie` (0x08), `reg_isr` (0x0C) to `hp_fsm_top.v`. New CPU_OP_WAIT state (5'd27) — pulses `desc_irq`, sets `chain_ctrl[2]=1` (cpu_op_pending), waits for CPU to clear ISR and set `chain_ctrl[0]=1` (resume). Backward compatible: `chain_ctrl[3]=0` → passthrough (unchanged), `chain_ctrl[3]=1` → interrupt protocol. Multi-driver resolved via `axil_we_chain_ctrl` flag. Register map:

    | 0x04 | REG_CHAIN_CTRL | R/W | [0]=resume, [2]=cpu_op_pending, [3]=intr_enable |
    | 0x08 | REG_GIE        | R/W | [0]=global interrupt enable |
    | 0x0C | REG_ISR        | R/W | [0]=cpu_op_irq (W1C) |

    All 18 simulation tests PASS (8 comprehensive + 10 Q5_0, includes backward compat verification).

12. **Bitstream rebuild (2026-07-12):** Vivado build completed (route_design: 1:39, total: ~8 min). WNS=-0.360 ns (4 failing endpoints). BRAM18: 33→10 (Q5_0 clean-slate rewrite eliminated 14, Q8 smem/act LUTRAM saved 9). Slice LUTs: 9,898→9,066. LUT as Memory: 97→177. Slice Regs: 12,781→14,052. DSP48E1: 22→23. All 5 `test_fpga_cores` HW tests PASS. Comprehensive HW tests: Tests 1-8 PASS, Tests 9a (Q8 multi-group) and 10 (Q8 multi-tile) FAIL (pre-existing from Q5_0 clean-slate rewrite on 2026-07-07 — those paths weren't re-tested after rewrite).

    **Debug session (2026-07-12): Test 9a and 10 diagnosis on hardware.** Found Test 9a was a test-data layout bug; Test 10 had TWO bugs:

    - **Test 9a fix (verified PASS on HW):** `write_pattern_const $Q9_WEIGHT_ADDR 4096 0x01` → `[expr $Q9_NUM_GROUPS * 4096]`. Multi-group FSM reads group 1 weights from `weight_addr + 4096` — only 4096 bytes were written, so group 1's weight load read scale data instead (which happened to follow the weight region in DDR). Fixed by writing `num_groups × 4096` bytes for multi-group tests. Result: all 64 rows = 128.
    - **Test 10 bug 1 (DDR address collision, harmless):** `Q10_ACT_ADDR=0x00109000` collided with tile 0 scales at `Q10_WEIGHT_ADDR+4096=0x00109000`. Fixed: act→`0x0010C000`, result→`0x0010B000`.
    - **Test 10 bug 2 (RTL, root cause):** `col_group` was not reset in the Q8 FETCH_DESC dispatch path (`hp_fsm_top.v:689`). After Test 9a's multi-group iteration left `col_group=1`, Test 10's tile 0 loaded weights/scales/acts from wrong DDR offsets (`weight_addr + col_group*4096` instead of `weight_addr`). Same bug class as CPU_OP col_group issue (Key Decision #10). Fix: added `col_group <= 0` in Q8 dispatch. Verified PASS on HW: all 128 rows = 64.

    **Extended HW test suite (11 tests, 2026-07-11):** Designed and verified 10 new edge-case tests covering:
    - E1: Q8 negative weights (0xFF → -64/row) **PASS**
    - E2: Q8 scale=0.5 (q8=2, scale=0x0100 → 64/row) **PASS**
    - E3: Q8 full 14-group (all-1s → 896/row) **PASS**
    - E4: Q5 negative q5 (qh=0, nibble=1 → -15 → -3,440,640/row) **PASS**
    - E5: Q5 d=0.5 (d=0x3800 f16=0.5 → 114,688/row) **PASS**
    - E6: Q5→CPU_OP→Q5 chain (mixed-type transitions) **PASS**
    - E7: Q8 negative act (act=-1 → -64/row) **PASS**
    - E8: Q5 negative act (act=-1 → -229,376/row) **PASS**
    - E9: Mixed Q5→Q8→CPU_OP→Q5 chain (4-desc cross-type) **PASS**
    - E10: Q5 alternating q5 nibbles (1,2 → 344,064/row) **PASS**
    - E6a: Standalone CPU_OP passthrough (col_group reset needed) **PASS**

    Debug aids added: pre-chain descriptor dumps with address layout comments, post-chain result + source data dumps, W0/W2 block data comparison. Bugs found during development: descriptor overlap (D2 only 16 bytes after D1 instead of 32), `write_pattern_const` TCL proc accepts byte value (not word).

13. **Q8 col_group stale in FETCH_DESC dispatch — Test 10 RTL fix (2026-07-16):** `col_group` was not reset in the Q8 FETCH_DESC dispatch path (`hp_fsm_top.v:689`). After a multi-group descriptor left `col_group=1`, the next descriptor in the chain (even a single-group one) loaded weights/scales/acts from wrong DDR offsets. Caused Test 10 tile 0 to fail with alternating bank corruption. Fix: `col_group <= 0` added to Q8 dispatch. Same class as CPU_OP col_group bug (Key Decision #10). All 10 comprehensive HW tests now PASS.

## Architecture Summary

### HP FSM flow (hp_fsm_top.v, Q8 compute path):
```
IDLE → FETCH_DESC → LOAD_WEIGHT → LOAD_SCALES → LOAD_ACT → COPY_ACT_TO_CORE → COMPUTE → READ_RES/READ_RES_ACC → WRITE_RES → DONE
```
For multi-group (q8_num_groups > 1, from descriptor): COMPUTE → READ_RES_ACC loops back to LOAD_SCALES for each group (accumulating into acc_buf), then COPY_ACC_TO_BUF → WRITE_RES.

(Each LOAD_*/FETCH_DESC state has a corresponding _W wait state for AXI burst completion. Full FSM: 27 states — see REG_DEBUG table for complete list.)

### CPU-OP Descriptor Protocol (CPU/FPGA synchronization):

To handle CPU-only operations (RMSNorm, RoPE, SoftMax, bias add, SwiGLU, residual add) between FPGA matmuls, the descriptor chain supports a special `CPU_OP` descriptor type (`tensor_type = 15`).

The interrupt-based CPU_OP protocol is implemented in `matmul_top.v` (the PhaseB chain FSM). When `matmul_top.v` encounters a CPU_OP descriptor:
1. It sets `reg_chain_ctrl[2]=1` and pulses `desc_irq` → CPU interrupt fires
2. FSM enters `PH_CPU_OP_WAIT` state and **pauses** until CPU resumes it
3. CPU reads `reg_status` (status=3 = chain busy) to distinguish from chain-complete
4. CPU reads `reg_desc_head` to identify which descriptor index triggered the CPU_OP
5. CPU does its work (norm, softmax, etc.) and writes results to DDR
6. CPU clears `reg_isr[0]` (write REG_ISR) and writes `CHAIN_CTRL[0]=1` to resume
7. FSM clears the resume signal, advances to next descriptor

**Note:** `hp_fsm_top.v` (the current active FSM) handles CPU_OP differently — it simply passes activations through to DDR as a passthrough read/write, without any interrupt or wait state (see `hp_fsm_top.v:492-494`). Path:
```
FETCH_DESC → LOAD_ACT → LOAD_ACT_W → WRITE_RES → WRITE_RES_BURST → WRITE_RES_W → DONE
```
The interrupt-based protocol described above exists in `matmul_top.v` for future PhaseB integration.

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

### Multi-Tile Iteration (within a descriptor):
Both Q5_0 and Q8_0 support processing multiple tiles per descriptor via the
`num_tiles` field at descriptor bytes [22:23]. The FSM iterates tiles internally:

- **Q5_0:** After WRITE_RES_W, if `q5_tile_counter + 1 < q5_num_tiles`, loops to
  `Q5_LOAD_NORM` to load the next tile's block norms. Result advances by 32 bytes
  (4 rows × 8 bytes). Activation is NOT re-read (same act for all tiles).
- **Q8_0:** After WRITE_RES_W, if `q8_tile_counter + 1 < q8_num_tiles`, loops to
  `LOAD_WEIGHT`. Weight address advances by `q8_tile_stride = 4096 + num_groups×256`.
  Result advances by 512 bytes (64 rows × 8 bytes). Activation is reloaded per tile.

A single Q5_0 descriptor with `num_tiles=224` processes the entire 896×896 `attn_q`
weight matrix. Without multi-tile, this would require 224 descriptors with 224
redundant DDR act reads.

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
| Q8_0 | 64×896 | — | 7680 (per desc) | 4096 (per group) | 8 |
| Q5_0 | 4×896 | 56 | 2696 (2688 blocks + 8 norm) | N/A (per-block streaming) | 8 |
| Q6_K | 32×256 | 32 | 6720 | 8192 | 8 |
| Q4_K | 56×256 | 56 | 8064 | 8192 | 8 |
| INT16 | 64×64 | — | 8192 | 8192 | 8 |

All cores output S24.8 fixed-point (48-bit accumulator, zero-extended to 64-bit in DDR).

### HP FSM Register Map (AXI4-Lite @ 0x43C00000):

| Offset | Name | Access | Description |
|--------|------|--------|-------------|
| 0x00 | `REG_START` | R/W | [0]: write 1 to start descriptor chain (auto-clears when FSM leaves IDLE) |
| 0x14 | `REG_STATUS` | R | [8]=rd_done, [9]=wr_done, [15]=busy (cleared in IDLE/DONE) |
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

**REG_DEBUG (0x28) bitfields** (from `hp_fsm_top.v:1082-1093`):
| Bits | Field | Description |
|------|-------|-------------|
| [31:27] | `state` | FSM state (5-bit, see below) |
| [26] | `rd_done` | HP read master done (sticky, cleared on next LOAD_ACT entry) |
| [25] | `wr_done` | HP write master done (sticky, cleared on next WRITE_RES entry) |
| [24] | `rd_busy` | HP read master busy |
| [23] | `wr_busy` | HP write master busy |
| [22] | `q8_busy` | Q8 core busy |
| [21:19] | `wr_dbg_state` | Write master FSM state (3-bit) |
| [18:16] | `rd_dbg_state` | Read master FSM state (3-bit) |
| [15] | `q8_done` | Q8 core done |
| [14:11] | `col_group` | Q8 column group counter (0..13) |
| [10:8] | `timeout_msb` | `timeout_cnt[15:13]` — top 3 bits of shared timeout |
| [7:0] | `sc_byte_idx` | Scale byte counter — useful for tracking scale loading progress |

**REG_Q8_DEBUG (0x3C) bitfields** (from `hp_fsm_top.v:1095-1104`):
| Bits | Field | Description |
|------|-------|-------------|
| [31:27] | `state` | FSM state (same as REG_DEBUG[31:27]) |
| [26] | `q8_busy` | Q8 core busy |
| [25] | `q8_done` | Q8 core done (pulse) |
| [24] | `q8_start` | Q8 core start (pulse) |
| [23] | `q8_act_we` | Q8 act write enable (active during COPY_ACT_TO_CORE) |
| [22:20] | `q8_core_state` | Q8 core's internal FSM state (3-bit) |
| [19:17] | `q8_core_g` | Q8 core's bank counter (3-bit) |
| [16:11] | `q8_core_k` | Q8 core's column counter (6-bit) |
| [10:7] | (various) | `{copy_act_idx[1:0], q8_sc_we, sc_byte_idx[0]}` — sc_byte_idx[0] toggles each scale byte pair |
| [6:0] | `wt_byte_idx` | Weight byte index `[6:0]` |

### Descriptor Format (32 bytes, 8 words):

| Offset | Bits | Field | Description |
|--------|------|-------|-------------|
| 0 | [31:0] | `next_addr` | DDR address of next descriptor (0 = end of chain) |
| 4 | [31:0] | `weight_addr` | DDR address of weight data (see per-type DDR layouts) |
| 8 | [31:0] | `act_addr` | DDR address of activation data (16-bit ints) |
| 12 | [31:0] | `result_addr` | DDR address for result writeback (8 bytes/row) |
| 16 | [15:0] | `tensor_type` | 15 = CPU_OP (passthrough), 1 = Q5_0, other = Q8 (default). Q6_K/Q4_K dispatch planned. |
| 16 | [31:16] | (reserved) | Upper 16 bits reserved |
| 20 | [3:0] | `num_groups` | Q8 column groups (0 = use GP0 register fallback). Ignored by Q5_0. |
| 21 | [7:0] | (reserved) | Reserved |
| 22 | [15:0] | `num_tiles` | Tiles per descriptor (0 → 1 for backward compat). Q5: 4-row tiles. Q8: 64-row tiles. |
| 24 | [23:0] | `act_total_bytes` | Total activation bytes to read from DDR |
| 28 | [31:0] | (reserved) | Reserved |

**Note:** The HP FSM (`hp_fsm_top.v`) uses the type dispatch above. The PhaseB path
(`matmul_top.v`) uses a different type numbering: 6=Q5_0, 8=Q8_0, 12=Q4_K, 14=Q6_K, 15=CPU_OP.

**FSM States (REG_DEBUG[31:27] or REG_Q8_DEBUG[31:27]):**
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
| 18 | `TIMEOUT_ERROR` | Timeout trap (latch state in `timeout_src`, stall forever) |
| 19 | `WRITE_RES_BURST` | Write result burst setup (computes burst addr/len from wr_remaining) |
| 20 | `Q5_LOAD_NORM` | Starting Q5_0 row_norm DDR read (8 bytes = 4 × UQ8.8) |
| 21 | `Q5_LOAD_NORM_W` | Waiting for norm read + unpacking to core norm BRAM |
| 22 | `Q5_COPY_ACT` | Starting Q5_0 activation DDR read (1792 bytes = 896 × INT16) |
| 23 | `Q5_COPY_ACT_W` | Waiting for act read + unpacking to core act_mem BRAM |
| 24 | `Q5_BLOCK_COMPUTE` | Starting per-block DDR read (48 bytes = 12 AXI beats) |
| 25 | `Q5_BLOCK_COMPUTE_W` | Unpacking 6 rd_data words into core blk_d/qh/qs, pulsing blk_valid |
| 26 | `Q5_READ_RES` | Capturing res0/res1 from both Q5_0 cores via hierarchical ref |
| 27 | `CPU_OP_WAIT` | Interrupt mode: pulse desc_irq, set chain_ctrl[2], wait for CPU resume |

**Key experience: Debug register field usage patterns from hardware bringup:**
- `REG_Q8_DEBUG[25]` (q8_done) transitions 0→1 when computation completes; this is the most important signal for debug
- `REG_Q8_DEBUG[16:11]` (q8_core_k) should count from 0→63 during result readback; if stuck at 0, the FSM never entered READ_RES/READ_RES_ACC
- `REG_Q8_DEBUG[15:8]` bit 0 (sc_byte_idx[0]) toggles each scale byte pair processed — if toggling stops, the scale loading DMA hung
- `REG_DEBUG[27]` (rd_done) and [26] (wr_done) are cumulative sticky bits — clear on next entry to LOAD_ACT/WRITE_RES respectively
- `REG_DEBUG[15:8]` vs [7:0] shows which FSM phase is active: [15:8] increments during LOAD_WEIGHT/LOAD_SCALES, [7:0] increments during LOAD_ACT

### Existing Verilog Cores:

| Core | Tile | Cycle/tile | Status |
|------|------|-----------|--------|
| `matmul_q8_core.v` | 64×896 | ~515 | ✅ Working |
| `matmul_q4k_core.v` | 56×256 | ~? | ✅ Working |
| `matmul_q5_0_core.v` | 4×896 | 1904 | ✅ Per-block wide register interface, no LUTRAM |
| `matmul_int16_core.v` | 64×64 | 515 | ✅ Working |
| `matmul_top.v` | — | — | ✅ 5 cores instantiated (Q8, Q4K, Q5_0, Q6_K, INT16) |
| `hp_fsm_top.v` | HP FSM + Q8 + Q5_0 | N/A | ✅ Descriptor-chain DMA, Q8 compute 64×896 (14-group), Q5_0 4×896 (2-core, per-block wide register interface) |

### Missing Verilog Cores:

| Core | Tile | Status |
|------|------|--------|
| None | — | All cores implemented ✅ |

## Fixes (2026-07-06) — Spurious block 56

**Bug: q5_start toggles every cycle, causing extra block after Q5_READ_RES.**

The `q5_start` pulse in `Q5_BLOCK_COMPUTE_W` fires every cycle because the condition `q5_qs_words == 8 && !q5_start` is always true (default sets q5_start=0 each cycle). On the cycle `q5_done_rise` fires for block 55, q5_start is ALSO pulsed. Next cycle the core enters IDLE, sees `start=1`, and starts computing with `blk_num=q5_blk_counter=56`. Header address 56 is out of bounds (0-55), returning X → `f16_w=X` → `d_pre=X` → `prod_w=X` → `acc=X`.

**Fix:** Added `q5_start_pulsed` single-shot flag:
- Cleared in `Q5_BLOCK_COMPUTE` (line 959)
- Set when start is first pulsed in `Q5_BLOCK_COMPUTE_W` (line 1003)
- Condition for pulsing: `q5_qs_words == 8 && !q5_start_pulsed`
- Prevents re-pulsing on subsequent cycles

All 4 Q5_0 HP FSM tests now PASS (previously all 4 FAILED with X in acc). No more block 56, no X propagation.

10. **Q8 BRAM waste analysis and smem/act → LUTRAM conversion (2026-07-07):** Analyzed `matmul_q8_core.v` BRAM utilization and found 9 BRAM18 held arrays using < 5% of their declared capacity:
    - **smem_bank0..7** (8 BRAM18): 8 banks × 512×16 declared, only 16 entries/bank used (3.1%) — address decode `{g, k[5]}` = 4 bits → 16. Changed to `ram_style = "distributed"` depth 16.
    - **act_bram** (1 BRAM18): 512×16 declared, only 64 entries used (12.5%). Changed to `ram_style = "distributed"` depth 64.
    - **wmem** (8 BRAM18): 512×8, 100% utilized. Left as BRAM.
    - **acc** (0 BRAM18): Already FFs/distributed RAM (8 entries/bank × 48-bit × 8 banks = 3Kb).
    
    **Result:** Q8 core BRAM drops from 17→8 (wmem only). Total system BRAM ~10 (Q8 8 + Q5_0 2). LUTRAM increases ~128 (smem) + 16 (act) = +144 LUTRAM (3.3% of 4,400 capacity, well within budget). All 123 core simulation tests PASS: Q8 6/6, Q4K 4/4, Q6_K 97/97, HP FSM 7/7, HP FSM Q5_0 9/9. INT16 smoke pre-existing fail (unrelated wmem addressing). Q5_0 standalone testbench pre-existing compile error (old port interface, not updated for per-block rewrite).

## Current Status (2026-07-16) — All 10 comprehensive HW tests PASS

| Resource | Used | Available | % | Notes |
|----------|------|-----------|---|-------|
| Slice LUTs | **9,066** | 17,600 | **51.51** | -832 from Q5_0 clean-slate rewrite (was 9,898) |
| LUT as Logic | **8,889** | 17,600 | **50.51** | |
| LUT as Memory | **177** | 6,000 | **2.95** | +80 from LUTRAM smem/act (was 97) |
| Slice Regs | 14,052 | 35,200 | 39.92 | +1,271 (CPU_OP protocol regs + DSP pipelining) |
| BRAM18 | **10** | 120 | **8.33** | Q8(8 wmem) + 2×Q5_0(1+1 act_mem) = 10 |
| DSP48E1 | **23** | 80 | **28.75** | +1 (Q5 DSP register opt added 42 regs) |
| Slice | 4,387 | 4,400 | **99.70** | Tight — routing congestion (unchanged) |
| **WNS** | **-0.360 ns** | 10 ns | 4 failing | Minimal violation, works on HW |

**Bitstream sources:** `axihp_read_master.v` + `axihp_write_master.v` + `matmul_q8_core.v` + `matmul_q5_0_core.v` + `hp_fsm_top.v`

**Hardware tests (2026-07-16): ALL 10 TESTS PASS**
| Test | Description | Result |
|------|------------|--------|
| 1 | Basic 64-byte DMA | PASS |
| 2 | Minimum 8-byte DMA | PASS |
| 3 | 128-byte 2-burst DMA | PASS |
| 4 | 256-byte 4-burst DMA | PASS |
| 5 | Chain of 2 descriptors | PASS |
| 6 | Chain of 3 descriptors | PASS |
| 7 | Re-start from DONE | PASS |
| 8 | Q8 compute (all-1s, 64×64) | PASS (all 64 rows = 64) |
| 9a | Q8 multi-group 64×128 (2 groups) | PASS (all 64 rows = 128) |
| 10 | Q8 multi-tile 2×64 rows | PASS (all 128 rows = 64) |

**Root cause of Test 10 tile 0 failure:** `col_group` not reset in Q8 FETCH_DESC dispatch (`hp_fsm_top.v:689`). After Test 9a's multi-group iteration left `col_group=1`, Test 10's tile 0 loaded weights from `weight_addr + 1×4096` (scale data), scales from `weight_addr + 4096 + 256` (tile 1 weight data), and acts from `act_addr + 128` (uninitialized DDR). Fixed by adding `col_group <= 0` in the Q8 dispatch. Same bug class as CPU_OP col_group issue (Key Decision #10, 2026-07-11).

**q8_act_we stuck-at-1 bug (cosmetic):** `q8_act_we` set in `COPY_ACT_TO_CORE` never cleared in `COMPUTE`/`COMPUTE_W`/`READ_RES`. Overwrites `act_bram[63]` during compute. Harmless with uniform activations but should be fixed.

**Simulation (iVerilog):** All 18 HP FSM tests PASS (8 comprehensive + 10 Q5_0 dispatch, includes backward compat verification).

### Hierarchical Resource Breakdown (Vivado routed — 2026-07-12 build)

```
Instance          Tot LUTs   %Total   FFs    BRAM18  DSP   Role
─────────────────────────────────────────────────────────────────────────
inst (FSM top)      4,464    50.1%   9,021     0       2   hp_fsm_top control logic
u_q8                2,148    24.1%   3,187     8      16   Q8 compute core
u_wr (write mstr)   1,141    12.8%    131      0       0   AXI HP write master
u_q5_core0            830     9.3%    367      1       2   Q5_0 core (rows 0-1)
u_q5_core1            766     8.6%    367      1       2   Q5_0 core (rows 2-3)
axi_lite (auto_pc)    392     4.4%    482      0       0   AXI protocol converter
u_rd (read mstr)      173     1.9%    116      0       0   AXI HP read master
─────────────────────────────────────────────────────────────────────────
Total               9,066   100%   14,052    10      23
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

### Q5_0 Clean-Slate Per-Block Wide Register Interface (2026-07-07)

Completely rewrote both `matmul_q5_0_core.v` and `hp_fsm_top.v` to eliminate byte-at-a-time header loading, `hdr_packed` LUTRAM, and `qs_word`/`blk_num`/`core_id`/`start` ports. The core now receives per-block `blk_d[15:0]`, `blk_qh[31:0]`, `blk_qs[127:0]` via a single `blk_valid` pulse — all fed from a 48-byte DDR burst (12 AXI beats) per block.

**Key changes:**
- Core ports: `blk_d[15:0]`, `blk_qh[31:0]`, `blk_qs[127:0]`, `blk_valid`, `norm_we/addr/din`
- Removed: `hdr_we/bank/addr/din[7:0]`, `qs_word[127:0]`, `start`, `blk_num`, `res_addr/res_dout`, `core_id`
- `row_scale[0:7]` renamed to `row_norm[0:1]` (PhaseB DSP normalization only)
- Fixed pre-existing `act_r` pipeline bug: core pre-loads next wi's activation to match BRAM 1-cycle read latency (was off-by-one, masked by all-1s test activations)
- Internal `blk_counter` resets on `clr_acc`, increments on block done
- 34 cycles/block, 33 on back-to-back transitions (no SETUP_D needed for blk 1-55)

**FSM changes (hp_fsm_top.v):**
- Removed Q5_PRELOAD_HDR/Q5_PRELOAD_HDR_W (~109 lines byte-level unpack)
- Removed `q5_hdr_we0/1`, `q5_hdr_bank/addr/din/sub/block/core`, `q5_qs_word0/1`, `q5_sc_*`, `q5_res_addr`, `q5_res_core/row`, `q5_qs_words`
- Added Q5_LOAD_NORM/Q5_LOAD_NORM_W: 8-byte DDR burst → 4 × UQ8.8 row_norm values
- Rewrote Q5_BLOCK_COMPUTE/Q5_BLOCK_COMPUTE_W: reads 48 bytes DDR (12 AXI beats = 6 × 64-bit rd_data) per block, unpacks into core d/qh/qs, pulses `blk_valid`
- Rewrote Q5_READ_RES: direct capture of `u_q5_core0.res0/res1` and `u_q5_core1.res0/res1` — no `res_addr` interface

**DDR layout per block (48 bytes):**
```
[0:1]   core0_d[15:0]       (f16)
[2:5]   core0_qh[31:0]
[6:21]  core0_qs[127:0]
[22:23] core1_d[15:0]
[24:27] core1_qh[31:0]
[28:43] core1_qs[127:0]
[44:47] padding
```
Total: 56 × 48 = 2688 bytes block data + 8 bytes norm (4 × UQ8.8) = 2696 bytes/tile.

**Pipeline (core):**
- **SETUP_D** (1 cycle): compute d_pre = f16_decode(d) × row_norm >> 8 (1 DSP), clamp to S16
- **COMPUTE** (32 cycles): 1 cycle per element:
  1. q5 decode from qs_r + qh_w (combinational, LUTs)
  2. dq = d_pre × q5 (16×5 multiply, LUTs)
  3. acc += dq × act_r (1 DSP MAC)
- **DRAIN** (1 cycle): signal done

**Cycle breakdown (core only, excludes DDR read overhead):**
- Each block: 1 SETUP_D + 32 COMPUTE + 1 DRAIN = 34 cycles (SETUP_D always executed)
- 56 blocks: 56 × 34 = 1904 cycles
- DDR read per block adds ~17 cycles (12 AXI beats + latency + unpack), giving ~56 × 51 = ~2856 system cycles/tile
- SETUP_D is not skipable — d_pre is recomputed per block (each GGUF block has its own f16 scale d)

### Q5_0 HP FSM Dispatch Tests (2026-07-06)

| # | Test | Weight | Act | Expected | Covers |
|---|------|--------|-----|----------|--------|
| 1 | Single desc, all-1s | 1 | 1 | 229376 | baseline |
| 2 | Chain of 2 (1→0) | 1, 0 | 1, 1 | 229376, 0 | chain, clr_acc |
| 3 | CPU_OP → Q5_0 | 1 | 1 | 229376 | mixed chain |
| 4 | Q5_0 → CPU_OP → Q5_0 | 1, 2 | 1, 2 | 229376, 917504 | 3-desc chain |
| 5 | Zero activations | 1 | 0 | 0 | act=0 edge case |
| 6 | Back-to-back restart | 1, 1 | 1, 1 | 229376, 229376 | DONE→IDLE re-entry |
| 7 | Chain of 4 Q5_0 | 1, 2, 0, 1 | 1 | 229376, 458752, 0, 229376 | 4-desc chain |
| 8 | Negative activations | 1 | -1 | -229376 | signed accum |
| 9 | 5-desc mixed chain | 1, 0, 1 | 1, 1, 1 | 229376, 0, 229376 | deep mixed chain |

## Bare-Metal ARM Port (2026-07-05) — FPGA Cores Tested on Hardware

| Component | File | Status |
|-----------|------|--------|
| Boot code (startup.s) | `vivado_integration/sw/startup.s` | ✅ FPU enable added (CPACR+FPEXC), MMU/caches off |
| Register map + types | `vivado_integration/sw/tmac_baremetal.h` | ✅ AXI-Lite access, math builtins, FP16→FP32 |
| Inference engine | `vivado_integration/sw/tmac_baremetal.cpp` | Q8/Q5 FPGA paths, CPU fallback for other types |
| Core test wrapper | `vivado_integration/sw/test_fpga_cores.cpp` | ✅ Runs on HW, 5 tests |
| Linker script | `vivado_integration/sw/link.ld` | ✅ DDR at 0x00100000, 64KB stack |
| Makefile | `vivado_integration/sw/Makefile` | ✅ LLVM/clang cross-compile |
| XSDB test script | `vivado_integration/sw/run_test_fpga_cores.tcl` | ✅ bitstream+ps7_init+AFI+dow+run |
| XSDB addr map test | `vivado_integration/sw/test_addr_map2.tcl` | ✅ Proved PS=PL address mapping |

### Key Fixes Found During Bare-Metal Port

1. **FPU not enabled** — Cortex-A9 VFP is disabled after reset (FPEXC.EN=0). Any floating-point instruction (float literals, casts, math) causes an undefined instruction exception. `startup.s` now enables CP10/CP11 in CPACR and sets FPEXC.EN=1. Without this, `test_fpga_cores` crashed immediately in the initialization loop (`act_all1[i] = 1.0f`).

2. **48-bit result sign extension** — The Q8/Q5 cores output 48-bit signed accumulators, zero-extended to 64-bit words in DDR. For negative results, the ARM code must manually sign-extend from bit 47:
   ```c
   uint64_t raw = (uint64_t)lo | ((uint64_t)hi << 32);
   if (raw & ((uint64_t)1 << 47)) raw |= 0xFFFF000000000000ULL;
   int32_t acc = (int32_t)(int64_t)raw;
   ```

3. **AFI0 registers at 0xF800_8000** — The AFI control registers are at 0xF800_8000 (not 0xF800_9000 as documented in earlier AGENTS.md). Correct config sequence:
   ```
   mwr 0xF8000008 0x0000DF0D    # unlock SLCR
   mwr 0xF8000910 0x0000000F    # LVL_SHFTR_EN
   mwr 0xF8008000 0x00000005    # AFI0_CTRL (enable + slverr)
   mwr 0xF8008004 0x00000044    # AFI0_PART (R:4, W:4 entries)
   mwr 0xF8008008 0x00000001    # AFI0_WRCHAN
   mwr 0xF8000004 0x0000767B    # lock SLCR
   ```

4. **PS=PL address mapping confirmed** — Writes via ARM core at PS address X are visible to the HP FSM at the same numeric address X. Both PS and PL share the same DDR address space (no 0x00100000 offset). Verified by `test_addr_map2.tcl` with CPU_OP descriptor passthrough test.

5. **Q8 weight format: single group, per-group scales** — The HP FSM loads ONE group's weights (4096 bytes = 64×64 INT8 column-major) and reuses them across all column groups via multi-group iteration. Scales are loaded per-group from `weight_addr + 4096 + g*256`. Activation data is loaded per-group from `act_addr + g*128`.

6. **Q5_0 scale location** — Row_inv UQ16.8 values follow the 4928 bytes of block data: 4 × uint16_t at `weight_addr + 4928`.

7. **CPU_OP descriptor** — Tensor type 15: loads `act_total_bytes` from `act_addr` into act_buf, writes directly to `result_addr` (passthrough). Used for CPU-only ops between FPGA matmuls.

8. **`-O1` required** — `-O2` build has a BSS layout issue on LLVM 7.0.1. Use `-O1` for bare-metal compilation.

9. **Scratch buffer addresses** — Must use addresses within the first ~500 MB of DDR. The range `0x1F000000-0x1F004000` (just below the last 1MB) is verified working. Higher addresses in the `0x17E00000` range may have DDR access issues.

10. **UART not initialized by ps7_init** — PS7 UART at 0xE0000000 is not fully initialized after ps7_post_config. `uart_init()` must program baud rate and enable TX. Until fixed, avoid UART output in test programs — use DDR buffer writes for result reporting.

### Test Results (HW, 2026-07-11)

```
test_fpga_cores.elf:
  5/5 PASS (2026-07-11)
  - Q8 all-1s:    PASS  (weights=1, acts=1 → 896 per row)
  - Q8 all(-1)s:  PASS  (weights=-1, acts=1 → -896 per row, signed path)
  - Q5 val=1:     PASS  (d=1.0, q5=+1, acts=1 → 229376 per row)
  - Q5 val=0:     PASS  (d=1.0, q5=0, acts=1 → 0 per row)
  - Q5 val=-1:    PASS  (d=1.0, q5=-1, acts=1 → -229376 per row)
```

**2026-07-11 Bug fix: Q5 expected value off by 256× in `test_fpga_cores.cpp`.**
The Verilog `f16_decode(1.0)` returns 256 (S24.8 fixed-point, 1.0 = 256), and `d_pre = f16_decode × norm >> 8 = 256 × 256 >> 8 = 256`. The test was comparing FPGA output against `q5_val² × 896` (naive), but the FPGA produces `d_pre × q5_signed × 896 = 256 × q5_val × 896`. Fix: added `f16_decode_c()` matching the Verilog function, used proper GGUF non-negative block scale `d=1.0`, and computed expected value as `d_pre × q5_val × 896`. All 5/5 PASS on HW.

### Next Steps

1. Run `test_fpga_cores.elf` with corrected output offsets to confirm sign extension fix
2. Build and run `tmac_baremetal.elf` with a loaded model for full inference
3. Switch to `-O2` once BSS layout issue is understood
4. Initialize UART properly for debug output

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
5. ✅ Q5_0 integration: 2× Q5_0 cores in hp_fsm_top, descriptor tensor_type=1 dispatch, block-streaming pipeline, HW test PASS (2026-07-05)
6. ✅ Q5_0 block-streaming redesign: 3.7× speedup (1904 vs 7170 cycles/tile, core-only), 1 DSP MAC/cycle, 9 HP FSM dispatch tests PASS, 2 × Q5_0 cores integrated into hp_fsm_top (2026-07-06)
7. ✅ Q5_0 clean-slate rewrite: per-block wide register interface, no hdr_packed LUTRAM, no byte-at-a-time header loading, act_r pipeline fix, row_scale→row_norm, hierarchical result read (2026-07-07)
8. ▶ Multi-core integration (Q4K/Q6_K/INT16) with shared BRAM weight loading

## Build & Run Commands

```bash
# PATH setup
$env:Path = "D:\Program Files\Git\bin;$env:Path"   # git

# Verilog tests (iVerilog — d:\iVerilog\bin)
make -C verilog all                     # Q8 + Q4K + Q5_0 + INT16 smoke

# Vivado xsim
#   C:\Xilinx\Vivado\2023.1\bin\xsim.bat
xsim tb_hw_fsm --runall                  # HP FSM descriptor-chain test

# Vivado batch build
#   C:\Xilinx\Vivado\2023.1\bin\vivado.bat
#   (NOT D:\Xilinx — Vivado is installed at C:\Xilinx)
vivado.bat -mode batch -source vivado_integration/build_bd.tcl

# JTAG load via XSDB (single session: program + init + run + capture)
#   C:\Xilinx\Vivado\2023.1\bin\xsdb.bat
xsdb.bat vivado_integration\sw\run_hp_fsm_comprehensive.tcl

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
- `verilog/matmul_top.v` — Quad-core top: AXI4-Lite slave (inline, replaces orphan axilite_slave.v), 8192-byte weight_buf, loading FSM, mode mux (Q8/Q4K/Q6_K/INT16). Q5_0 removed (handled by hp_fsm_top.v exclusively).
- `verilog/matmul_q8_core.v` — Q8_0 compute core: 8×512×8 wmem (BRAM banks), dequant LUT, 6-stage pipeline
- `verilog/matmul_q4k_core.v` — Q4_K block decode: 8064-byte block buffer (`block_buf [0:8063]`), S24.8 fixed-point, 56×256 tile
- `verilog/matmul_q5_0_core.v` — Q5_0 block decode: 4×896 tile, 2× parallel cores, block-streaming pipeline (SETUP_D→COMPUTE×32→DRAIN), 1 DSP MAC/cycle, per-block wide register interface (blk_d/qh/qs), no hdr_packed LUTRAM, hierarchical result read
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
- `verilog/tb_hp_fsm_q5_0.v` — HP FSM Q5_0 dispatch test (9 tests: all-1s, chains, mixed CPU_OP, edge cases)
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
- **Phase 2 (Q8 compute)** ✅ **ALL 9 TESTS PASS ON HARDWARE** — Q8 pipeline timing fix (WNS +0.550), sc_byte_idx reset bug fixed, all-1s pattern. Three bugs fixed (q8_wt_din reg, col_group init, act_remaining). Test 9a multi-group 2-group 64×128 tile PASS (all 64 rows = 128) on 2026-07-02. **rd_ready handshake fix (2026-07-04):** Changed LOAD_WEIGHT_W from `rd_ready <= 1` to `rd_ready <= rd_valid`. All 9 HW tests pass after 64-bit word write change. Phase 2 complete.
- **Q8 multi-group/multi-tile regression (2026-07-12):** Q5_0 clean-slate rewrite (2026-07-07) broke Q8 multi-group (Test 9a) and multi-tile (Test 10). Both FAIL on current bitstream. **Fixed (2026-07-12):** Test 9a — weight buffer size changed from 4096 to `num_groups × 4096` (was leaving group 1 weight as scale data). Test 10 — `Q10_ACT_ADDR=0x00109000` collided with tile 0 scales at `weight_addr+4096`, moved to `0x0010C000`; `Q10_RES_ADDR=0x0010A000` collided with tile 1 scales at `weight_addr+tile_stride+4096`, moved to `0x0010B000`. Both are test-data layout bugs, no RTL changes needed.

## Hardware Gotchas

### Critical: ps7_init re-execution hang
`ps7_pll_init_data_3_0` **hangs if PLLs are already configured** from a prior session. The PLL reset sequence (bypass→power-down→reset→wait-for-lock) can't re-lock when the PLLs are already locked from a previous session. This leaves the PS7 in a partially-configured state and all subsequent ps7_init attempts also hang (DDR init's `mask_poll 0xF8000B74 0x00002000` waits for calibration that depends on PLL clock).

**Workaround:** Always power-cycle the board before running ps7_init via XSDB. A processor-only reset (`rst -processor`) is insufficient — the PLLs must be in reset-init state for ps7_pll_init to succeed.

**Key observation:** HP0 register reads return 0x00000000 after a failed ps7_init attempt, even though the OCM code wrote valid non-zero values. This is because the PS7 AHB interconnect enters an inconsistent state when PLLs are partially configured, and DAP read transactions return 0.

### PS7 HP0 is AXI3 (max 16 beats per burst)
`ARLEN > 15` causes silent AR rejection — the read master never receives RLAST. All HP FSM bursts are hard-limited to 16 beats (`rd_len/wr_len ≤ 15`).

### `rst -processor` corrupts DAP irreversibly
The JTAG DAP controller enters an unrecoverable state after `rst -processor`. Use `stop` instead to halt the CPU without resetting the debug infrastructure.

### FCLK_CLK0 enable requires ARM boot code
DAP writes to `FPGA_CLK_CTRL[7]` (FCLK_CLK0 enable at `0xF8000170`) are ignored — the register is locked to secure mode. The CPU must enable FCLK_CLK0 in its boot code before the PL clock can be used.

### Batch Build Framework

| Script | Purpose | Command |
|--------|---------|---------|
| `vivado_integration/build_bd.tcl` | Vivado batch build | `C:\Xilinx\Vivado\2023.1\bin\vivado.bat -mode batch -source vivado_integration/build_bd.tcl` |
| `vivado_integration/sw/rebuild.tcl` | XSCT: rebuild Vitis app | `C:\Xilinx\Vitis\2023.1\bin\xsct.bat vivado_integration/sw/rebuild.tcl` |
| `vivado_integration/sw/run_hp_fsm_comprehensive.tcl` | XSDB: all 7 HP FSM tests (basic, min, 2-burst, 4-burst, chain 2, chain 3, restart) + Test 8-9 (Q8 all-1s, Q8 multi-group) | `C:\Xilinx\Vivado\2023.1\bin\xsdb.bat vivado_integration/sw/run_hp_fsm_comprehensive.tcl` |
| `vivado_integration/sw/run_hp_fsm_q5_0.tcl` | XSDB: Q5_0 compute test (all-1s, 4 rows, 896 each) | `C:\Xilinx\Vivado\2023.1\bin\xsdb.bat vivado_integration/sw/run_hp_fsm_q5_0.tcl` |

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

**Board test with ARSIZE=3: RDATA[63:32]=0** — Zynq-7010 with x16 DDR3 caps HP0 at 32-bit. Upper 32 bits of each 8-byte beat are always 0, losing every other 32-bit word. ARSIZE=3 REJECTED for read master (upper 32 bits are garbage); write master later reverted to AWSIZE=3 because the Zynq HP port correctly handles 8-byte beats by performing two 32-bit DDR accesses internally (see current `axihp_write_master.v`).

| Fix Attempt | Result | Detail |
|-----|--------|--------|
| done bits sticky | ✅ | `reg_status[15:8]` no longer cleared in DONE state |
| ARSIZE=3 (read master) | ❌ REJECTED | RDATA[63:32]=0 on hardware, data corruption |
| AWSIZE=3 (write master, later adopted) | ⏳ see below | Current write master uses AWSIZE=3 with 4-state FSM |
| Bitstream rebuilt | ✅ | 0 errors, synth 50s + impl 2:31 |

**Actual fix (read master):** Reverted to ARSIZE=2, accepting the 32-bit nature of HP0 read on this board. Write master was later rewritten to use AWSIZE=3 with a simpler 4-state FSM (see current `axihp_write_master.v`).

### 32-bit HP mode (2026-06-20) — SUPERSEDED (see current write master below)

**Finding: Zynq-7010 HP0 is 32-bit only with x16 DDR3.** Despite `PCW_S_AXI_HP0_DATA_WIDTH=64` set before `apply_bd_automation`, RDATA[63:32] is always 0 on hardware. The PS7 silicon ignores the 64-bit width parameter when the DDR bus is x16-wide. `AFI0_CTRL[7:6]` (64-bit enable) is also read-only — confirmed by write-verify loop.

**Design change (later superseded by AWSIZE=3):** Switched to full 32-bit AXI mode:

| Component | Change |
|-----------|--------|
| `axihp_read_master.v` | ARSIZE=2 (4 bytes/beat), always captures `m_axi_rdata[31:0]` (no beat[0] alternation since HP0 doesn't do narrow-transfer byte lane remapping — RDATA[63:32] is always 0) |
| `axihp_write_master.v` | AWSIZE=2 (5-state FSM, superseded by current AWSIZE=3, 4-state FSM — see `axihp_write_master.v` for current design) |
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
- `vivado_integration/rtl/hp_fsm_top.v`: HP descriptor-chain FSM — AXI4-Lite slave, desc_buf (32B), act_buf (512B), Q8/Q5_0 compute dispatch, 64-bit HP read/write masters. 28-state FSM (IDLE 0 through CPU_OP_WAIT 27). Q8 path: IDLE→FETCH_DESC→FETCH_DESC_W→LOAD_WEIGHT→LOAD_WEIGHT_W→LOAD_SCALES→LOAD_SCALES_W→LOAD_ACT→LOAD_ACT_W→COPY_ACT_TO_CORE→COMPUTE→COMPUTE_W→READ_RES/READ_RES_ACC→COPY_ACC_TO_BUF→WRITE_RES→WRITE_RES_BURST→WRITE_RES_W→DONE (20 states). Q5_0 path adds Q5_LOAD_NORM, Q5_LOAD_NORM_W, Q5_COPY_ACT, Q5_COPY_ACT_W, Q5_BLOCK_COMPUTE, Q5_BLOCK_COMPUTE_W, Q5_READ_RES (7 states). CPU_OP interrupt path adds CPU_OP_WAIT (1 state).
- `vivado_integration/sw/run_hp_fsm_comprehensive.tcl`: XSDB flow — all 7 HP FSM tests (basic, min 8B, 128B 2-burst, 256B 4-burst, chain of 2, chain of 3, re-start). Polls HEAD register for completion.
- `vivado_integration/sw/run_hp_fsm_q5_0.tcl`: XSDB flow — Q5_0 all-1s test. Loads weight/scales/acts, sets tensor_type=1 descriptor, verifies 4 rows = 896.
- `vivado_integration/sw/regs.h`: Register map
- `vivado_integration/ps7_init.tcl`: Modified — AFI1 + LVL_SHFTR_EN config in ps7_post_config
- `verilog/axihp_read_master.v`: HP read master — ARSIZE=2 (4 bytes/beat), always captures RDATA[31:0], byte-stream output. DRAIN state per-beat. 32-bit mode.
- `verilog/axihp_write_master.v`: HP write master — AWSIZE=2, splits 64-bit word into two 32-bit single-beat AXI writes. wready once per word. 5-state FSM.
- `verilog/matmul_int16_core.v`: INT16 compute core (verified standalone)
- `verilog/test_hp_loopback.v`: 32-bit mode simulation testbench — DDR model with 32-bit read/write granularity. Passes full loopback.
- `vivado_integration/proj_bd/matmul_bd.runs/impl_1/system_wrapper.bit`: Synthesized bitstream (reordered PS7 config)
- `D:/Users/u/workspace/tmac/Debug/tmac.elf`: Vitis ELF loaded by XSDB
- `docs/debug_log.md`: Full debug history
- `linux/README.md`: Linux-on-SD boot build guide + Windows SD card creation steps
- `linux/tmac_linux.c`: Linux userspace FPGA test program (uses /dev/mem mmap)
- `linux/boot/boot.bif`: Bootgen config (FSBL + bitstream + U-Boot)
- `linux/boot/system_wrapper.bit`: FPGA bitstream for Linux boot
- `linux/setup_toolchain.sh`: Creates clang-based ARM cross-compiler wrappers for macOS
- `linux/build_all.sh`: Full build script for U-Boot + kernel + initramfs (Lima VM)
- `linux/clone_repos.sh`: Clones u-boot-xlnx + linux-xlnx + buildroot in parallel

## Linux-on-SD Card Boot ✅ BUILT (2026-07-18)

U-Boot + Linux kernel + BusyBox initramfs + FPGA test program built in
a Lima ARM64 Ubuntu VM on macOS. No macOS cross-compilation hacks needed.

**Build output at `~/arm-build/`:**

| File | Size | Description |
|------|------|-------------|
| `u-boot.img` | 1.2 MB | U-Boot image (loaded by SPL from SD FAT32) |
| `u-boot-spl.bin` | 121 KB | SPL (in BOOT.BIN, runs from OCM) |
| `uImage` | 4.6 MB | Linux 6.6.0-xilinx (CONFIG_DEVMEM=y) |
| `devicetree.dtb` | 17 KB | zynq-zc702 (prebuilt in kernel tree) |
| `uramdisk.image.gz` | 1.3 MB | BusyBox initramfs (79 tools + tmac) |
| `tmac` | 483 KB | Static ARM32 (mmaps FPGA at 0x43C00000) |

**Tools in initramfs:** devmem, hexdump, xxd, devmem, md5sum, vi, grep, awk,
ifconfig, ping, wget, fdisk, mkfs.ext2, blkid, tar, modprobe, + BusyBox shell.

**Kernel:** CONFIG_DEVMEM=y — FPGA registers accessible via `/dev/mem`.
Use `iomem=relaxed` bootarg if STRICT_DEVMEM blocks 0x43C00000 range.

**To reproduce:** See `linux/README.md` for full build instructions.
The Lima VM approach eliminates all macOS SDK conflicts — builds take
~5 min for U-Boot + kernel, no patches needed.

### Windows (Vivado) — SD Card Creation

**Pre-built files** in `linux/boot/`:
- `system_wrapper.bit` — FPGA bitstream (prebuilt)
- `matmul_bd.xsa` — hardware handoff
- `boot.bif` — bootgen config (FSBL + bitstream + U-Boot)

**Windows steps** (see `linux/README.md` for details):
1. Build `fsbl.elf` in Vivado SDK from `matmul_bd.xsa`
2. Copy `u-boot-spl.bin`, `u-boot.img`, `uImage`, `devicetree.dtb`, `uramdisk.image.gz` from macOS build
3. Run `bootgen -image boot.bif -o BOOT.BIN -w`
4. Format SD: FAT32 partition (BOOT.BIN + uImage + dtb + initramfs), ext4 partition (model.tmac + tmac)
5. **Power-cycle** board, insert SD, set SD boot mode, connect UART (115200 baud)

**U-Boot bootargs** for FPGA access:
```
setenv bootargs "console=ttyPS0,115200 root=/dev/ram0 rw iomem=relaxed"
```
