# Q5/Q8 Core Optimization Plan

Status: ACTIVE. Started 2026-08-17.

## Goal & priorities

- **Priority: throughput over resource reduction.** Make the Q5_0 and Q8_0
  compute paths faster first; use resource reduction only where it directly
  enables more parallelism (frees slices for wider datapaths).
- **End goal (for now): make Q5/Q8 faster/leaner.** Re-fitting the Q4_K/Q6_K
  cores is explicitly OUT of scope at this moment.
- **Hard constraint:** PL slice utilization is ~97.8% (4,303/4,400) and is the
  binding wall. LUTs 53%, DSP 23/80 (29%), BRAM 10/120 (8%) all have headroom.

## Baseline facts (from RTL + 2026-08-17 resource report)

| | Q8 core | Q5_0 core (x2) |
|---|---|---|
| Tile | 64x64 per group, 14 groups | 2 rows x 896 |
| MACs/cycle | 8 (act-mul DSP) + 8 (dequant DSP) | 1 (single MAC DSP) |
| Pipeline | 6-stage PRE->S0->S1a->S1b->S2a->S2b | SETUP_D..D4 + COMPUTE x32 + DRAIN |
| Cycles/tile (core only) | ~515/group (x14) | 56 blocks x 37 = 2072 |
| Resources | 2,148 LUT, 3,187 FF, 8 BRAM, 16 DSP | ~830 LUT, 367 FF, 1 BRAM, 2 DSP each |

Key insight: DSP headroom (29% -> up to ~60% feasible) is the throughput lever;
FF/control-set count is the slice lever.

## Phase 0 - Line-by-line RTL walkthrough (COMPLETE)

Deliverable: `docs/q5_q8_core_analysis.md` - a per-file line-by-line explanation
with a cycle-by-cycle pipeline trace.

1. `matmul_q8_core.v` (661 lines): memory system (wmem/smem/act/acc banks),
   the (g,k) iteration order, the address-anticipation scheme, CLEAR_ACC
   arming, the 4 DRAIN states, `dequant_q8` fixed-point, `res_dout` readback.
2. `matmul_q5_0_core.v` (351 lines): `blk_valid` block streaming, 28-blocks-per-row
   layout, `f16_decode` (S24.16, sign-fixed) -> `d_pre` (S16) -> `dq` (LUT 16x5)
   -> DSP MAC -> S48 acc, the `act_r` BRAM pre-load + wi_preload/wi_for_prod
   pairing, the `clr_acc` DSP flush.
3. `hp_fsm_top.v` interaction: descriptor -> load -> compute -> readback
   sequencing, DDR burst sizes, the single-flight read master, and where cycles
   are lost to non-overlapped DMA.

Also resolve the Q8 DSP discrepancy (header says 8, resource table says 16) and
pin the exact per-tensor cycle budget.

## Phase 1 - Measurement baseline (IN PROGRESS)

- Cycle-count each tensor's compute-vs-DDR-load split (REG_CLK_CNT + sim).
- Refresh the per-module Vivado resource + timing report for the current build.
- Confirm the top consumers (almost certainly the 896x896 / 4864x896 Q5 tensors
  and the serialized DDR loads).

Status: resource + timing refreshed (see `q5_q8_core_analysis.md` section 7.1);
analytical cycle model written (7.2) - headline finding is **Q5 is ~98% of the
FPGA cycle budget**; `--cycles` instrumentation added to `tmac_linux.c` and tmac
rebuilt (md5 0467e91b...) for the on-board confirmation run (7.3).

## Phase 2 - Optimization (re-ranked by Phase 1 measurement)

Phase 1 found Q5 is ~99.7% of FPGA cycles and 72% DDR-bound, so the priority is
**Q5 first**, Q8 deprioritized. Revised ranking:

**1. Q5 block-DMA overlap + batching** (biggest win). The 48-B block read is
serialized with the 37-cycle block compute, and each tiny read pays full DDR
latency. Overlap block g+1's read with block g's compute, and/or batch several
blocks per burst. Target: 7,510 -> ~2,072 cyc/tile (~3.6x) before any MAC change.

**2. Q5 MAC parallelism** (raise the compute floor). 1 MAC/cycle today. 2-4
elements/cycle (multiple q5 decodes + DSPs, multi-bank act_mem) shrinks the
2,072-cycle compute floor 2-4x. Uses DSP headroom (23/80).

**3. Q5 SETUP_D overhead removal**. 4 cycles/block (~11%) for f16_decode + d_pre;
a register-pipelined block header double-buffer hides it.

**4. Q8 dequant -> LUTs**. Frees 8 DSPs for Q5/Q8 widening; low throughput impact
(Q8 is 0.3%) but cheap DSP headroom.

**5. Reduce slice pressure** (enables 1-4):
- D1: FSM-top act_buf/acc_buf/desc_buf FF arrays -> BRAM (~7,400 FFs, 1 RAMB18, 0 LUTs).
- D2: Q8 acc banks -> LUTRAM (the "scattered RAM" cleanup, see
  `q5_q8_core_analysis.md` section 6): the banks are declared 512-deep (above the
  LUTRAM max of 64), so 3,072 FFs hold 8 live entries/bank. Re-declare depth 8,
  trim 48 -> ~33 bits (true sum bound), and consolidate the 3 read sites (RMW,
  res_dout, acc_r) to one port: ~3,072 FFs -> ~264-384 LUTs.
- D3: consolidate the 307 unique control sets.

**6. Timing**. WNS -0.209 (7 fail); widen with pipeline-depth headroom.

(Original A->B->C order, where B was Q8 widening, is superseded by the Phase 1
finding that Q8 is negligible.)

## Phase 3 - Verification gate (after each change)

iVerilog regression (extend with wider/non-uniform patterns) -> Vivado build +
slice/timing check (slice < ~100%) -> on-silicon `--compare`/`--trace` to
preserve bit-exactness.

## Sequencing

Phase 0 (doc) -> Phase 1 (baseline) -> Phase 2 item 1 (Q5 DMA overlap, Steps 1-3) ->
item 2 (Q5 MAC parallelism, Step 4) -> item 3 (SETUP_D removal, Step 5) ->
items 4-5 (Q8 dequant / slice incl. LUTRAM cleanup, Steps 6-7) -> item 6 (timing).

## Execution contract (2026-08-17)

STABLE, ONE STEP AT A TIME. Each step is small, self-contained, and only proceeds
after the previous step is CONFIRMED WORKING. Never batch multiple RTL changes.

Per-step gate (all must pass before the next step):
1. iVerilog regression green (tb_hw_fsm_comprehensive 10/10, tb_hp_fsm_q5_0 10/10,
   tb_matmul_q6_k 97/97, tb_q5_negd, Q8 6/6) - extended with cycle assertions
   where the step changes timing.
2. Vivado build: slice < ~100%, timing not materially worse.
3. Board: `--compare` bit-exactness preserved + `--cycles` shows the expected
   per-tile reduction.

## Phase 2 execution steps (Q5-first, gated)

| Step | Change | Target (cyc/tile) | Risk |
|------|--------|-------------------|------|
| 1 | Sim cycle-count gate in tb_hp_fsm_q5_0 (test-only, no RTL) | measure baseline 7,510 | none (test-only) |
| 2 | Q5 double-buffer block fetch (overlap block g+1 read with block g compute) | ~3,500 | moderate (FSM only) |
| 3 | Read-master burst chaining (back-to-back 16-beat bursts, no inter-burst wait) | ~2,200 | moderate (read master) |
| 4 | Q5 MAC parallelism (2-4 elements/cycle) | ~1,000 | higher (core datapath) |
| 5 | SETUP_D overhead removal (may fold into step 4) | - | low |
| 6 | Q8 dequant -> LUTs (frees 8 DSPs) | - | low |
| 7 | Q8 acc banks -> LUTRAM (depth 8 + one read port + 33-bit trim) | resource: -2.7k FFs (+~0.3k LUTs) | low-moderate (core) |

Each step stops after its gate passes; the plan is updated with the confirmed
result before the next step begins.

## Step results log

- **Step 1 (COMPLETE 2026-08-17):** sim cycle-count gate added to
  tb_hp_fsm_q5_0 (counts uut.reg_status[15] busy cycles; start_chain snapshots,
  wait_done reports). All 10 tests PASS. **Sim baseline: 5,736 cyc/single tile**
  (act load included), ~4,563 cyc/tile on multi-tile (act loaded once). Board
  baseline is 7,510 (real DDR latency); sim DDR model is idealized (1 cyc/beat),
  so sim is a relative gate, not an absolute one.

- **Step 2 (COMPLETE 2026-08-17):** Q5 double-buffer block fetch implemented in
  hp_fsm_top.v. The FSM now issues the block g+1 48-byte DDR read while block g
  computes (block g's data is latched by the cores on the 1-cycle blk_valid pulse,
  so the q5_blk_* regs are safely reused). Added
  q5_dispatched/q5_done_pending/q5_last_pulsed; Q5_BLOCK_COMPUTE issues only the
  first read; Q5_BLOCK_COMPUTE_W does the overlap dispatch.
  Sim: 5,736 -> 3,866 cyc/tile (1.48x), multi-tile 4,563 -> 2,693 (1.70x); all
  regressions PASS. Vivado: LUTs 9,831 (55.86%), slice 4,243 (96.43%, improved),
  WNS -0.425 (21 fail, pre-existing Q5-core path; accepted by decision).
  **Board (CONFIRMED): Q5 per-tile 7,510 -> ~5,300 cyc (1.42x)**; `--compare`
  bit-exact (no RAWDIFF); Q8 attn_v unchanged (~146,800). Per-token FPGA cycles
  ~528M -> ~374M (~5.3s -> ~3.7s @ 100 MHz). Bitstream B1E46B14 md5 / BOOT.BIN
  A588409D (deployed to SD).

- **Step 3 (COMPLETE 2026-08-17):** pipeline the Q5 block reads so the ~50-cycle
  DDR latency per 48-B burst is amortized into a continuous stream (the board was
  read-bound: per-block ~70 cyc = latency + 12 beats + 6 words, and 56 x 70 ~=
  the measured non-compute tile cost). Implemented in two coordinated changes
  (3a read master + 3b FSM), after one reverted attempt that exposed a subtle
  buffer-tracking bug (see below). Verified on silicon: every `[q5 acc=gold]` and
  `[q8 acc=gold]` bit-exact, no `[q8 RAWDIFF]`, generation starts `9370 3837`
  (matches sim greedy reference). Bitstream `A5C1505757285D013FF42839EC310B1F222
  BAB0A8764B1AB37E162767596A848` / BOOT.BIN `ED866A07...` (deployed to SD).

- **Step 3a (COMPLETE 2026-08-17):** pipelined AR support in `axihp_read_master.v`.
  A 1-deep start queue accepts `start_rise` while a burst drains; the next burst's
  AR is issued as soon as the AR channel is free (SEND_AR), or completes its
  handshake while the current burst drains in READ_BEAT/PRESENT, so consecutive
  bursts pipeline on the PS7 HP0. `done` pulses per burst (even when a queued
  burst is promoted), so per-burst consumers (Q5 blocks, Q8 weight bursts) are
  unchanged; single-flight callers (Q8, CPU_OP) are unaffected. New focused test
  `tb_read_master_pipe.v` PASSED.

- **Step 3b (COMPLETE 2026-08-17):** Q5 read-ahead + double-buffered unpack in
  hp_fsm_top.v. Added buffer B (q5_blkB_*), explicit per-block buffer tracking
  (`q5_unpack_blk` follows the block index being unpacked, NOT a free-running
  flip - this is the fix over the reverted attempt below), an arm/pulse dispatch
  (`q5_pulse_arm`/`q5_pulse_buf`), and read-ahead issuing gated on `!rd_start`
  (the block-0 issue in Q5_BLOCK_COMPUTE leaves `rd_start` high for one cycle;
  without the gap the block-1 issue collides and is silently dropped, shifting the
  pipeline and stalling on block 55). Two correctness guards:
  - **Read-ahead gate** `!rd_start` prevents a dropped `start` when the master is
    still busy with the just-issued burst.
  - **In-order dispatch guard** (`q5_next_expected = dispatched+1`): the Q5 core
    addresses activations by its internal pulse-order `blk_counter` (`act_blk =
    row_high ? blk_counter-28 : blk_counter`, `act_mem[0:1023]`), so blocks MUST
    pulse strictly 0..55. The old A-first preference dispatched block N+2 (in A)
    before block N+1 (in B) once reads ran ahead, multiplying block N+2's weights
    against the wrong activation segment (invisible under all-1s tests, exposed by
    21-run in-order tracing in sim). Only the buffer holding `q5_next_expected`
    may dispatch.

  **Sim verification:** 21 runs x 56 blocks strictly in-order; `tb_hp_fsm_q5_0`
  10/10 + `tb_hw_fsm_comprehensive` 10/10 + `tb_matmul_q8` 6/6 + `tb_q5_negd`
  ALL PASS; no `$display` debug left in the RTL.
  **Vivado:** first build WNS -0.751 (263 fail) -> recovered via
  `vivado_integration/improve_timing.tcl` (Performance_Explore + ExtraTimingOpt
  place + AggressiveExplore route + phys_opt) to **WNS -0.343, 6 failing
  endpoints** (comparable to the documented working -0.209/7 build; critical path
  is the pre-existing Q5-core `wi_reg -> act_mem -> q5 -> acc` route). Slice
  4,349 (98.84%).

- **Step 3 first attempt (ATTEMPTED + REVERTED 2026-08-17):** an earlier 3b used
  a `q5_unpack_buf` flip to select the unpack target. **BUG FOUND in sim:** the
  flip desynchronized from the block-to-buffer assignment (`q5_blk_counter%2`)
  partway through a tile - block 54's data unpacked into the wrong buffer, the
  last block never dispatched, and the FSM stalled. Reverted to the verified
  Step 2 state. **Lesson applied:** track the unpack buffer explicitly per block
  index (as the final 3b does with `q5_unpack_blk`) rather than a free-running
  flip.

- **D1 (ATTEMPTED + REVERTED 2026-08-17):** FSM buffers -> LUTRAM. Tried the
  simple approach (`ram_style = "distributed"` hint on act_buf/acc_buf + serialize
  Q5_READ_RES's 4 simultaneous writes). **FAILED:** Vivado ignored the hint (FF
  count unchanged at 14,076, LUT-as-Memory unchanged at 177) because the access
  pattern is not single-write-port / single-read-port (act_buf has 4 write
  sources, acc_buf has 2 read addresses) - distributed-RAM inference needs ONE
  muxed write port + ONE muxed read port. The build even regressed (slice 4,243
  -> 4,310, timing 7 -> 53 failing) from the added serialization logic at the
  congested edge. Reverted. **A proper conversion needs an explicit single-port
  restructure (mux all write/read sources into one port each) - moderate effort,
  and AGENTS.md already flags D1 as ~50-70 slices ("not the bottleneck": the
  slice limit is the 307 control sets + carry/mux, not FFs).**

## Decisions log

- 2026-08-17: priority = throughput; end goal = faster/leaner Q5/Q8 (not
  re-fitting Q4/Q6).
- 2026-08-17 (Phase 1 result): Q5 is ~99.7% of FPGA cycles and 72% DDR-bound;
  re-ranked Phase 2 to Q5-first, Q8 widening deprioritized (0.3% impact).
- 2026-08-17 (Step 3 attempt): Q5 read-ahead + read-master pipelining attempted
  and REVERTED - the double-buffer unpack-target flip desynced mid-tile in sim.
  Cleaner buffer/target tracking needed before re-attempting.
- 2026-08-17 (Step 3 re-attempt, COMPLETE): unpack target tracked per block index
  (`q5_unpack_blk`) instead of a free-running flip; read-ahead gated on `!rd_start`;
  in-order dispatch guard added after out-of-order pulses (block N+2 before N+1)
  were seen once reads ran ahead. All sim regressions PASS, board `--compare`
  bit-exact, generation `9370 3837`. Bitstream A5C15057 (WNS -0.343 via
  improve_timing.tcl) / BOOT.BIN ED866A07 deployed.
