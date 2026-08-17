# Q5_0 / Q8_0 Core Line-by-Line Analysis

Phase 0 deliverable of `q5_q8_optimization_plan.md`. A per-file walkthrough of the
two compute cores and their FSM interaction, with a cycle-level model. This is the
shared mental model for all subsequent optimization work.

---

## 1. Shared terminology & number formats

- **tile**: one weight-matrix slice processed by one descriptor.
  - Q8: 64 rows x 896 cols (14 groups x 64 cols). Multi-group = 64x64 per group.
  - Q5: 4 rows x 896 cols (2 cores x 2 rows). 224 tiles cover attn_q (896x896).
- **group (Q8)**: a 64x64 column slab. `g` indexes a row-bank (0..7), `k` a column (0..63).
- **block (Q5)**: 32 consecutive elements of one row. 28 blocks/row, 56/tile.
- **lane (Q8)**: one of the 8 parallel weight bytes / accumulators in a cycle.

Fixed-point conventions (see `docs/maths.md` for notation):
- Q8 scale `sc` = UQ8.8 (1.0 = 256). Dequant = `q8 * sc >> 8` -> INT16.
- Q8 accumulator = S48 (signed 48-bit).
- Q5 block scale `d` = f16. Decoded by `f16_decode` to **S24.16** (1.0 = 65536).
- Q5 `d_pre` = `f16_decode(d) * norm >> 8`, clamped to **S16** (1.0 = 256).
- Q5 `dq` = `d_pre * q5` (LUT 16x5), S21. Q5 acc = S48.

---

## 2. `matmul_q8_core.v` (661 lines)

### 2.1 Ports (lines 10-41)

Compute control: `start`, `op_vecmul` (unused in current path), `done`, `busy`.
Memory load ports (driven by the FSM): `wt_we/wt_addr[8:0]/wt_din[63:0]` (weight),
`sc_we/sc_addr[6:0]/sc_din[15:0]` (scale), `act_we/act_addr[5:0]/act_din[15:0]`
(activation). Result: `res_addr[5:0]` -> `res_dout[47:0]` (combinational). The
`dbg_*` outputs (lines 28-40) are ILA probes added 2026-08-16; they are plain
`assign` of existing signals and are unconnected in the non-ILA build.

### 2.2 Memory system

| Array | Style | Depth x width | Contents |
|-------|-------|---------------|----------|
| `wmem_bank0..7` (73-80) | BRAM18 | 512 x 8 | one byte lane each; `{g,k}` -> byte for (row 8g+lane, col k) |
| `smem_bank0..7` (102-109) | LUTRAM | 16 x 16 | per-lane scale; `{g,k[5]}` -> scale for that 32-col half-block |
| `act_bram` (133) | LUTRAM | 64 x 16 | activation per column `k` |
| `acc_b0..7` (146-153) | dist RAM/FF | 512 x 48 | per-lane accumulator; only entries `[g]` (0..7) are live |

- Weight write (82-93): one 64-bit DDR word is broadcast: bank `i` gets byte `i`
  (`wt_din[i*8+7 : i*8]`). The FSM writes sequential words, so `wt_addr` is the
  word index. The DDR layout is row-group-major (see AGENTS.md "Q8 DDR Layout"):
  word `w = g*64 + k` holds rows 8g..8g+7 of column k.
- Scale write (115-128): `sc_addr[3:1]` selects the bank, `{sc_addr[6:4], sc_addr[0]}`
  the local address. The FSM streams 256 bytes -> 128 x 16-bit scales.
- Activation write (135-138): `act_bram[act_addr] <= act_din`, 64 entries.
- Accumulator: 8 banks, each holding 8 live entries `acc_bX[g]` = accumulator for
  row `g*8 + X`. Declared 512 deep but only the 3-bit `g` index is used. The 512
  depth exceeds the LUTRAM max (64), so Vivado builds it as FFs (a major FF cost,
  see 5); re-declaring depth 8 + consolidating read ports would let it become LUTRAM.

### 2.3 Pipeline stages (6-stage, lines 207-231)

`PRE -> S0 -> S1a -> S1b -> S2a -> S2b`

| Stage | Registers | Operation |
|-------|-----------|-----------|
| PRE | `pre_wmem_addr/pre_smem_addr/pre_act_addr`, `pre_g/pre_k/pre_valid` | compute next-cycle BRAM read addresses |
| S0 | `wmem_rdata[0:7]`, `smem_rdata[0:7]`, `act_rdata`, then `p1_*` | latch BRAM/LUTRAM outputs |
| S1a | `dq_deq[0:7]`, `dq_act`, `dq_row_base`, `dq_valid` | dequant: `dequant_q8(wmem, sc)` (8x in parallel) |
| S1b | `p2_partial[0:7]`, `p2_row_base`, `p2_valid` | `dq_act * dq_deq[wi]` (8x DSP MAC) |
| S2a | `pre_read_g`, `p2a_valid` | pre-read `acc_bX[g]` (1-cycle RAM latency) |
| S2b | (RMW in FSM) | `acc_bX[g] <= acc_bX[g] + p2_partial[X]` |

Note the dequant (S1a, an 8x16 multiply mapped to a DSP) and the MAC (S1b, a 16x16
DSP) are **separate DSP stages** -> 16 DSPs total (8 dequant + 8 MAC). The header
comment "8 DSPs" is stale.

### 2.4 FSM (lines 277-593)

States: `IDLE(0) CLEAR_ACC(1) COMPUTE(2) DRAIN(3) DRAIN2(4) DRAIN3(5) DRAIN4(6)`.

- **IDLE** (298-313): on `start`, set `busy`, zero `k/g/acc_clr_cnt`, go CLEAR_ACC.
- **CLEAR_ACC** (315-356): for 8 cycles, zero `acc_bX[acc_clr_cnt]` for all X. This
  walks the 8 live entries per bank. At `acc_clr_cnt==6` it *pre-arms* the first
  smem/act read addresses; at `==7` it advances to the second entry's addresses and
  sets `pre_valid`, entering COMPUTE. The 2-cycle-ahead anticipation for smem/act
  (vs 1-cycle for wmem) exists because smem/act pass through an extra Stage-0
  register.
- **COMPUTE** (358-454): the steady-state. Each cycle advances the 6-stage pipeline
  one step and performs one accumulation RMW (S2b). The counter `(g,k)` iterates
  g=0..7 inner, k=0..63 outer (lines 444-453): 512 iterations = 512 RMW cycles,
  8 MACs each = 4096 MACs = 64 rows x 64 cols.
- **DRAIN..DRAIN4** (456-590): flush the 5 remaining pipeline stages after the last
  COMPUTE entry, one accumulate per stage per cycle. `DRAIN4` sets `done=1`, `busy=0`.

### 2.5 Address anticipation (the 2026-08-12 bug)

`pre_wmem_addr/pre_smem_addr/pre_act_addr` (270-272) are registered in the COMPUTE
case (430-442), NOT in a separate always block (a separate block was optimized away
by Vivado, leaving addresses stuck at 0). The scheme:

- wmem is anticipated 1 iteration ahead: `{g+1, k}` (or `{0, k+1}` on wrap).
- smem/act are anticipated 2 iterations ahead because of the extra Stage-0 register:
  `{g+2, k[5]}` (smem) and `k` (act), with wrap cases at g==6 and g==7.

The BRAM reads (624-648) run every cycle with these addresses; the outputs are only
captured by the `p1_*` registers when `pre_valid`/`p1_valid` indicate validity.

### 2.6 Math

- `dequant_q8` (653-659): `(q8 * sc) >>> 8`, q8 INT8, sc UQ8.8 -> INT16. `$signed`
  on the product then arithmetic right-shift (rounds toward -inf, matches the golden
  model).
- MAC (382): `p2_partial[wi] = $signed(dq_act) * dq_deq[wi]` (INT16 x INT16 -> S32).
- Accumulate (361-368): `acc_bX[g] <= acc_bX[g] + p2_partial[X]` (S48).

### 2.7 Result readback

`res_dout` is combinational (175-188): `res_addr[2:0]` picks the bank, `res_addr[5:3]`
the group. This was made combinational to fix a 1-cycle registered-output off-by-one
in the FSM's READ_RES (AGENTS.md "Bug 4").

---

## 3. `matmul_q5_0_core.v` (351 lines)

One core = 2 rows. Two cores per tile (rows 0-1 and 2-3). Block-streaming: the FSM
pushes one block's `d/qh/qs` per `blk_valid` pulse; no weight storage in the core.

### 3.1 Ports (18-61)

- `core_id` selects rows 0-1 (id 0) vs 2-3 (id 1) for `norm_idx`.
- `norm_we/norm_addr[1:0]/norm_din[15:0]`: 4 per-row normalization values (UQ8.8),
  written once per tile, persist across blocks.
- `blk_d[15:0]/blk_qh[31:0]/blk_qs[127:0]/blk_valid`: one block's GGUF data.
- `act_we/act_addr[9:0]/act_din[15:0]`: 896 x INT16 activation BRAM, shared by both
  rows, pre-loaded by the FSM.
- `clr_acc`: clears accumulators and flushes DSP pipeline.
- `res0/res1` (S48) are combinational reads of the two accumulators.
- `dbg_*` (53-60): free-running debug visibility.

### 3.2 Block format & row tracking (115-119)

`blk_counter` 0..55; `row_high = blk_counter >= 28` selects row 0 (0..27) or row 1
(28..55). `act_blk = row_high ? blk_counter-28 : blk_counter` maps each row's 28
blocks onto the same 28 activation column-blocks (both rows read the same 896 acts).

### 3.3 f16_decode -> d_pre -> dq (166-203)

- `f16_decode` (166-183): converts f16 to **S24.16** (1.0 = 65536). The 2026-08-14
  sign fix applies bit 15 to negate the magnitude (llama.cpp's negative-d trick).
  Subnormals/denormals/exp 0/31 -> 0.
- `d_fp = f16_decode(blk_d_r)` (192).
- `norm_idx = {core_id, row_high}` (193) -> `row_norm[norm_idx]`.
- `d_pre_shr = d_pre_full_r >>> 8` (196), clamped to S16 (197-200). `d_pre_full_r`
  is the registered DSP product `d_fp * norm` (SETUP_D2, line 289), breaking a
  2-deep DSP cascade that was the 13.17 ns critical path.
- `dq = d_pre * q5` (203), a 16x5 LUT multiply -> S21. `dq_r` (79) registers it,
  breaking the 12.258 ns d_pre->q5->dq->DSP path.

### 3.4 q5 element decode (144-153)

From latched `blk_qs_r`/`blk_qh_r`: `qs_byte = blk_qs_r[wi_for_prod[3:0]*8 +: 8]`,
`ql` = low or high nibble by `wi_for_prod[4]`, `qh = blk_qh_r[wi_for_prod]`,
`q5 = {qh, ql} - 16` (5-bit signed, range -16..15). Pure LUTs, no multiplier.

### 3.5 Activation pre-load pipeline (121-142)

`act_mem` is a 1-read-port BRAM (1-cycle latency). The core pre-loads the *next*
element each cycle so the MAC always sees the current element in `act_r`:

- `blk_entry = blk_valid && (IDLE || DRAIN)`.
- `wi_preload` (136-141): 0 in SETUP_D/SETUP_D2/SETUP_D3/DRAIN-entry, `wi` in
  SETUP_D4, `wi+1` in COMPUTE (0 on wrap at wi==31).
- `act_addr_pre = act_blk*32 + wi_preload` (142).
- `wi_for_prod` (149): `wi` in SETUP_D4, else `wi_preload` — pairs the decoded q5
  with the BRAM-pipelined `act_r`.

The comment block (125-135) explains the off-by-one pairing in detail (the `dq_r`
pipeline adds 1 cycle, so COMPUTE wi=0 skips the stale `prod_r`, and wi=1..31 +
DRAIN accumulate the 32 correct elements).

### 3.6 FSM (236-338)

States: `IDLE(0) SETUP_D(1) SETUP_D2(2) SETUP_D3(3) SETUP_D4(4) COMPUTE(5) DRAIN(6)`.

- **IDLE** (272-281): on `blk_valid`, latch d/qh/qs, set busy, -> SETUP_D.
- **SETUP_D** (283-286): `d_fp_r <= d_fp` (register the long f16_decode path).
- **SETUP_D2** (288-292): `d_pre_full_r <= d_fp_r * norm_s` (DSP), wi=0.
- **SETUP_D3** (294-297): `d_pre <= d_pre_next` (shift + clamp).
- **SETUP_D4** (299-303): -> COMPUTE.
- **COMPUTE** (305-317): 32 cycles. `acc[row_high] += prod_r` (skip wi=0; the DSP
  pipeline adds 1 cycle so the first real product lands at wi=1). `prod_r` is the
  registered DSP output `dq_r * act_r` (211, 232).
- **DRAIN** (319-335): accumulate the last element, `done=1`, increment `blk_counter`
  (wraps 55->0), and if `blk_valid` is already asserted go back-to-back into
  SETUP_D (block streaming), else IDLE.

Per block: SETUP_D..D4 (4) + COMPUTE (32) + DRAIN (1) = 37 cycles. 56 blocks =
2072 cycles/tile for 2 rows x 896 = 1792 MACs -> ~0.86 MAC/cycle (the 4 SETUP_D
cycles per block are pure overhead).

### 3.7 DSP flush / clr_acc (213-233)

`clr_acc` is held by the FSM for 16 cycles (Q5_BLOCK_COMPUTE, hp_fsm_top.v 1086-1099)
so the DSP48E1's internal AREG/BREG/MREG/PREG all flush: `prod` is muxed to 0 and
`dq_r/prod_r/act_r` are zeroed, preventing 2-3 stale accumulations from pipeline
residue.

---

## 4. FSM interaction (`hp_fsm_top.v`)

### 4.1 Descriptor dispatch (705-772)

`FETCH_DESC_W` parses the 32-byte descriptor and branches: type 15 -> CPU_OP,
type 1 -> Q5 path (`Q5_LOAD_NORM`), else -> Q8 path (`LOAD_WEIGHT`). The Q8 dispatch
(line 762) now also clears `sc_burst_done` (the 2026-08-17 fix).

### 4.2 Q8 path

```
LOAD_WEIGHT -> LOAD_SCALES -> LOAD_ACT -> COPY_ACT_TO_CORE -> COMPUTE ->
READ_RES (or READ_RES_ACC for multi-group) -> [COPY_ACC_TO_BUF] -> WRITE_RES
```

- **LOAD_WEIGHT/LOAD_WEIGHT_W** (833-882): 4096 B weights in 64-B bursts (rd_len=15
  = 16 beats x 4 B). `wt_byte_idx[11:3]` = word address, direct 64-bit write to
  wmem. Multi-group reloads per group (`weight_addr + col_group*4096`).
- **LOAD_SCALES/LOAD_SCALES_W** (887-950): 256 B scales, unpacked 64-bit -> 4 x
  16-bit into smem via `rd_unpack_cnt`. Uses `sc_burst_done`/`rd_unpack_active`
  handshake (the source of the 2026-08-16 bug).
- **LOAD_ACT/LOAD_ACT_W** (775-828): activations into `act_buf` (128 B/group for
  Q8 = 64 x INT16), `act_addr + col_group*128`.
- **COPY_ACT_TO_CORE** (1181-1191): 64 cycles, `act_buf` -> `q8_act` 16-bit at a time.
- **COMPUTE/COMPUTE_W** (1194-1217): pulse `q8_start`, wait `q8_done_rise`.
- **READ_RES** (1220-1230, single group) / **READ_RES_ACC** (1233-1256, multi-group):
  read 64 x 48-bit results into `act_buf` (or accumulate into `acc_buf` across
  groups; group g loops back to LOAD_WEIGHT until all groups done).
- **WRITE_RES/WRITE_RES_BURST/WRITE_RES_W** (1272-1362): write 512 B (Q8) or 32 B
  (Q5) result to DDR; multi-tile loops back (Q5 -> Q5_LOAD_NORM, Q8 -> LOAD_WEIGHT
  with `weight_addr += tile_stride`).

### 4.3 Q5 path

```
Q5_LOAD_NORM -> [Q5_COPY_ACT] -> Q5_BLOCK_COMPUTE (56x) -> Q5_READ_RES -> WRITE_RES
```

- **Q5_LOAD_NORM** (974-1021): 8 B -> 4 x UQ8.8 `norm` (Q5_LOAD_NORM_W). Skip
  Q5_COPY_ACT on tile > 0 (act already resident).
- **Q5_COPY_ACT** (1023-1074): 1792 B -> `act_mem` (shared by both cores), 64-B
  bursts, unpacked 64-bit -> 4 x INT16.
- **Q5_BLOCK_COMPUTE/Q5_BLOCK_COMPUTE_W** (1076-1165): per block, a 48-B DDR read
  (12 beats = 6 x 64-bit), unpacked into `q5_blk_d0/d1, q5_blk_qh0/1, q5_blk_qs0/1`
  for both cores, then `blk_valid` pulse; `clr_acc` held 16 cycles on block 0;
  waits `q5_done_rise`.
- **Q5_READ_RES** (1167-1178): capture `res0/res1` from both cores into `act_buf`
  (4 x 48-bit).

### 4.4 DMA characteristics (the throughput limiter)

- The read master is **single-flight** (one burst in flight; a re-`rd_start` while
  busy is dropped) and **32-bit** (ARSIZE=2, 4 B/beat). Bursts are <= 16 beats.
- Load and compute are **fully serialized**: the FSM reads weights/scales/acts,
  then computes, then reads results, then writes results — with no overlap between
  DMA and compute. DDR latency is added directly to every group/tile.
- Q5 block streaming is also serialized: each 48-B block read waits for the read
  to complete before `blk_valid`; the 37-cycle block compute is not overlapped with
  the next block's 48-B read.

---

## 5. Cycle budget (approximate, core-only)

| Path | Per unit | Cycles |
|------|----------|--------|
| Q8 group (64x64) | CLEAR_ACC 8 + COMPUTE 512 + DRAIN 4 = ~524 | ~524 x 14 = ~7,336 / tile |
| Q8 per-group overhead | COPY_ACT 64 + READ_RES 64 | +128 x 14 |
| Q5 tile (4 rows x 896) | 56 x 37 | 2,072 |
| Q5 per-block overhead | SETUP_D..D4 = 4/block | 224 / tile |

DDR load time (not counted above) is the dominant real cost because it is not
overlapped. The 896x896 and 4864x896 Q5 tensors dominate the token latency (224
tiles and 1216 tiles respectively, each 2,072 core cycles + DDR).

## 6. Optimization observations (teed up for Phase 2)

1. **DMA overlap (A)** is the single biggest lever: the serialized load-compute
   sequence wastes most of the DDR bandwidth. Double-buffering weights/acts would
   hide it.
2. **Q8 dequant uses 8 DSPs** for 8x16 multiplies that could move to LUTs (or fold
   into the accumulator), freeing 8 DSPs for MAC widening (B).
3. **Q8 acc banks are FF-heavy** (8 x 48 x 8 = 3072 FFs, built as FFs because they
   are declared 512-deep, above the LUTRAM max of 64 - depth is the blocker, not the
   48-bit width). Re-declaring depth 8 (and trimming to ~33 bits, the true sum bound)
   + consolidating the three read sites to one port would make them LUTRAM:
   ~3,072 FFs -> ~384 LUTs (or ~264 at 33 bits) (D2).
4. **Q5 SETUP_D overhead** is 4 cycles per 32-element block (~11%). A
   register-pipelined block header (d/qh/qs double-buffer) could hide it and enable
   back-to-back blocks at 32 cycles/block.
5. **Q5 is 1 MAC/cycle** and latency-bound; 2-4 element parallelism (C) or a second
   act_mem read port is needed to raise throughput.

---

## 7. Phase 1 baseline (measurement)

### 7.1 Refreshed resource + timing (2026-08-17 routed build)

| Metric | Value |
|--------|-------|
| Total LUTs / logic / LUTRAM / SRL | 9,346 / 9,169 / 118 / 59 |
| FFs | 14,076 |
| BRAM18 | 10 |
| DSP48E1 | 23 |
| Slice | 4,303 (97.80%) |
| Setup / Hold | 7 failing, WNS -0.209 ns / 0 failing, +0.027 ns |

Note: the hierarchy is FLATTENED (`FLATTEN_HIERARCHY=full`), so the routed
checkpoint no longer separates `u_q8` / `u_q5_core0` / `u_q5_core1` / `u_wr` /
`u_rd`. `axi_hp_top` (all of hp_fsm_top) = 8,954 LUTs / 13,594 FFs / 10 BRAM /
23 DSP. To get a per-core breakdown again, re-synthesize with flatten off.

### 7.2 Analytical cycle model (compute exact, DDR estimated)

Read master: ARSIZE=2 (4 B/beat), combines 2 beats -> one 64-bit word, single
flight, <= 16 beats/burst. A 64-B burst ~= 1 (AR) + DDR_latency (~25) + 16
(beats) + 8 (word handshakes) ~= 50 cycles (DDR_latency is the estimate).

Per-tile cost (DDR + compute, serialized):

| Phase | Q8 (per group) | Q5 (per tile) |
|-------|----------------|---------------|
| weight / norm | 64 bursts x 50 = ~3,200 | 1 burst = ~30 |
| scales | 4 bursts = ~200 | - |
| act | 2 bursts = ~100 | 28 bursts = ~1,400 (tile 0 only) |
| block data | - | 56 x (12 beats ~30 + 37 compute) = ~3,750 |
| copy/readback | 64 + 64 = 128 | 1 |
| compute | ~524 | (inside block) |
| result write | (once/tile) ~400 | ~30 |
| **total** | **~4,150 / group (x14 = ~58k/tile)** | **~3,800-5,200 / tile** |

Per-layer tile counts (Q5 dominates):

| Tensor | Type | Tiles |
|--------|------|-------|
| attn_q | Q5 896x896 | 224 |
| attn_k | Q5 128x896 | 32 |
| attn_v | Q8 128x896 | 2 |
| attn_output | Q5 896x896 | 224 |
| ffn_gate | Q5 4864x896 | 1,216 |
| ffn_up | Q5 4864x896 | 1,216 |
| ffn_down | Q6/Q4 -> CPU | - |

Headline: **Q5 is ~98% of the FPGA cycle budget per layer** (~2,912 tiles x
~4,000 cyc = ~11.6 M cycles vs ~3 Q8 tiles x 58k = ~175k), because of (a) the
huge 896x896 / 4864x896 shapes, (b) 1 MAC/cycle, and (c) the serialized 48-B
block DMA that is not overlapped with the 37-cycle block compute.

### 7.3 Board measurement (CONFIRMED 2026-08-17)

`./tmac model.tmac --cycles` measured REG_CLK_CNT deltas per matmul (100 MHz):

| Tensor | Type | Tiles | Measured | Per tile |
|--------|------|-------|----------|----------|
| attn_q | Q5 896x896 | 224 | ~1,683,000 | 7,510 |
| attn_k | Q5 128x896 | 32 | ~240,000 | 7,510 |
| attn_v | Q8 128x896 | 2 | ~146,800 | 73,400 |
| attn_v | Q5 (12 layers) | 32 | ~240,000 | 7,510 |
| attn_output | Q5 896x896 | 224 | ~1,683,000 | 7,510 |
| ffn_gate | Q5 4864x896 | 1,216 | ~9,136,000 | 7,510 |
| ffn_up | Q5 4864x896 | 1,216 | ~9,135,000 | 7,510 |

Per-layer FPGA total ~= 22.0 M cycles; per token (24 layers) ~= 529 M cycles ~= 5.3 s
@ 100 MHz.

**Breakdown per layer:** ffn_gate + ffn_up = **83%**, attn_q + attn_output = 15%,
attn_k = 1.1%, attn_v = 0.7% (Q8 ~= 0.3%). **Q5 is ~99.7% of FPGA cycles.**

**Q5 compute-vs-DDR split:** 7,510 cyc/tile = 2,072 compute (56 blocks x 37, 28%)
+ ~5,440 DDR/overhead (72%). The 48-B block read is serialized with the 37-cycle
block compute, and its per-read DDR latency is not amortized.

**Q8:** 73,400 cyc/tile = 14 groups x ~5,240, dominated by the 4,096-B per-group
weight reload (64 bursts) serialized with ~524 compute cycles. Q8 is negligible
to total throughput (0.3%), so Q8 widening is deprioritized.

**Conclusion (flips the plan's A->B->C emphasis):** the Q5 path is the only thing
that matters, and it is 72% DDR-bound. The two highest-leverage changes are:
1. overlap/batch the Q5 48-B block reads (target: 7,510 -> ~2,072 cyc/tile, ~3.6x),
2. raise Q5 MACs/cycle (2-4 elements/cycle) to shrink the compute floor.


