# Debug Experiences & Toolbox

Practical, field-tested debugging techniques for this Zynq-7010 accelerator,
written from real sessions. The centerpiece is the first successful ILA
(Integrated Logic Analyzer) campaign (2026-08-16, the Q8 "first-run
corruption"), plus the iVerilog and host-side techniques that complement it.

Every technique here was used to find a real bug in this design:

| Technique | Bugs it found |
|-----------|---------------|
| iVerilog sim + non-uniform data | Q8 address-advancement, DDR weight layout, Q8 scale normalization |
| Bit-exact golden models | Q5 sign bit, S48->int32 wrap, res_dout off-by-one, Q5 d_pre precision |
| Host-side C byte-compare | Q6_K missing `+bo`, f16_to_f32 unsigned wrap |
| On-board `--compare`/`--trace` | Q6_K dequant bugs, layer-norm explosion |
| ILA (2026-08-16) | Q8 first-run corruption = stale `sc_burst_done` scale-burst skip |
| Testbench address-coverage audit | (unmasked the same stale-flag bug - see iVerilog section) |

---

## 1. ILA (Integrated Logic Analyzer) - the 2026-08-16 session

### 1.1 Why ILA was needed

The Q8 first-run corruption reproduced **only** on the board: after any Q5_0
descriptor, the *first* Q8 descriptor corrupts rows 16-63 (acc-bank groups
g=2..7); a re-run of the identical descriptor is clean. The existing debug
registers (`REG_DEBUG`, `REG_Q8_DEBUG`) are single 32-bit **snapshots** read
via AXI4-Lite. They tell you *which state* the FSM is in at the instant you
poll, but you cannot see the *causal sequence* of cycles that produced a
wrong result. Polling a live FSM also perturbs it (and the window is only
~874 cycles).

ILA gives cycle-accurate, non-invasive capture of a window of signals around
a chosen trigger - exactly what was needed to see the first Q8 COMPUTE's
CLEAR_ACC writes and accumulator RMW reads.

### 1.2 Design for observability (RTL side) - no logic change

The golden rule: **ILA probes must be existing internal wires; adding them
must not alter the design structure.**

1. **Expose core internals as `dbg_*` outputs** (`verilog/matmul_q8_core.v`
   lines 28-40, 54-63): `dbg_state`, `dbg_k`, `dbg_g`, `dbg_acc_clr_cnt`,
   `dbg_p2_valid`, `dbg_p2_row_base`, `dbg_acc_rw_rd` (combinational RMW read
   of `acc_b0`), `dbg_p2_partial0`, `dbg_pre_read_g`, `dbg_acc_r0`. These are
   `assign` from existing registers/wires - no behavioral change.

2. **The "do not probe what you must not change" rule.** `dbg_acc_r0`
   (the *dead* `acc_r` pre-read block) was deliberately **NOT** wired into the
   ILA bus: instantiating its read port would force that dead block live and
   change the acc-bank LUTRAM access structure, so the observed behavior would
   no longer match the deployed bitstream. Before probing a signal, ask:
   "does driving this to an output force logic that is currently
   synthesized-away to become real?"

3. **Trim wide values.** The 48-bit acc values were reduced to their low
   32 bits - enough to distinguish garbage from zero at the first RMW - to
   save probe bits.

4. **Pack into a single `dbg_bus[141:0]`** at the FSM top
   (`vivado_integration/rtl/hp_fsm_top.v` lines 114-130, 567-585). A single
   probe keeps the BD wiring trivial. Bit layout (document in a comment next
   to the port!):

   ```
   [0]      dbg_first_q8      (trigger, see below)
   [5:1]    FSM state
   [8:6]    q8_busy / q8_start / q8_done
   [12:9]   col_group
   [18:13]  q8_res_addr
   [50:19]  q8_res_dout[31:0]
   [53:51]  core dbg_state   [59:54] dbg_k   [62:60] dbg_g
   [67:63]  core acc_clr_cnt [68] p2_valid  [74:69] p2_row_base
   [106:75] core acc_rw_rd[31:0]   [138:107] p2_partial0[31:0]
   [141:139] core pre_read_g
   ```

5. **Build a software-triggerable, self-latching marker.** The whole bug is
   "first Q8 after a Q5", so instead of trying to trigger on a raw data
   pattern, the RTL synthesizes the event:

   ```verilog
   reg q5_seen;            // set when a Q5_0 descriptor compute completes
   reg dbg_q8_trig_done;   // latch so the pulse fires only once per boot
   if (state == Q5_READ_RES) q5_seen <= 1;
   if (dbg_first_q8)        dbg_q8_trig_done <= 1;
   wire dbg_first_q8 = (state == COMPUTE) && q5_seen && !dbg_q8_trig_done;
   ```

   This produces a single 1-cycle pulse exactly at the corrupt window entry.
   Triggering on a semantic event beats triggering on raw signal values every
   time.

### 1.3 Block-design instantiation (`vivado_integration/build_bd.tcl`)

```tcl
create_bd_cell -type ip -vlnv xilinx.com:ip:ila:6.2 ila_0
set_property -dict [list \
    CONFIG.C_NUM_OF_PROBES {1} \
    CONFIG.C_PROBE0_WIDTH {142} \
    CONFIG.C_DATA_DEPTH {1024} \
] [get_bd_cells ila_0]
connect_bd_net [get_bd_pins ps7/FCLK_CLK0] [get_bd_pins ila_0/clk]
connect_bd_net [get_bd_pins axi_hp_top/dbg_bus] [get_bd_pins ila_0/probe0]
```

Gotchas:

- **Clock the ILA with the SAME clock as the logic under test** (here
  `FCLK_CLK0`, the 100 MHz PL clock driving the FSM). The ILA samples its
  probe bus on this clock edge.
- **No manual `dbg_hub` cell.** The debug hub (JTAG BSCAN access to the ILA's
  S_AXI) is auto-inserted by Vivado during implementation for BD-debug cores.
  Manually adding a `dbg_hub` IP fails - the `dbg_hub:3.0` VLNV does not
  exist in Vivado 2023.1.
- **Probe width budget.** The first attempt packed a full 238-bit bus; it
  **failed placement**. Trim to 142 bits (drop wide values to 32-bit
  sub-ranges, drop the least-essential fields). More ILA width costs routing
  and LUTs - keep it lean. Verify the design still passes the iVerilog
  regression suite after adding the bus (the RTL carries the bus into sim too).

### 1.4 Build artifacts

After implementation, the probe file `linux/boot/system_wrapper_ila.ltx`
(LTX = Logic eXport) is generated. Because the bitstream changed, re-fuse
`BOOT.BIN` with `bootgen` (`bootgen.bat -image boot.bif -o BOOT.BIN -w` in
`linux/boot/`). The `.ltx` must be loaded by every runtime script or the ILA
probes will not resolve.

### 1.5 Runtime control - Hardware Manager TCL (`linux/scripts/*.tcl`)

Commands belong to **Vivado's Hardware Manager** (`vivado.bat -mode tcl`),
NOT xsdb.

The three-phase flow:

| Phase | Script | Notes |
|-------|--------|-------|
| Arm + wait + read (single session) | `arm_wait_read_ila_q8.tcl` | **The working pattern** (see below) |
| Read back an already-captured trace | `read_ila_q8.tcl` | Data persists in the FPGA until reprogrammed |
| Arm-then-disconnect (older) | `arm_ila_q8.tcl` | **Broken pattern** - closing the target disarms the ILA |

Critical runtime gotchas:

1. **Never close the JTAG target between arm and trigger.** The ILA is
   disarmed when the connection drops. `arm_ila_q8.tcl` (arm then
   `close_hw_target`) does NOT work reliably - `arm_wait_read_ila_q8.tcl`
   keeps the session open, polls, and dumps the CSV when the capture lands.
2. **Reprogram the PL between runs.** The `dbg_q8_trig_done` latch is
   once-per-boot; any prior `tmac` run has already latched it, so a second
   capture never triggers. The script does `program_hw_devices` first to
   reset the trigger logic.
3. **Trigger properties are split.** `CONTROL.TRIGGER_POSITION` and
   `TRIGGER_COMPARE_VALUE` are settable at runtime; `TRIGGER_MODE`,
   `TRIGGER_CONDITION`, `CAPTURE_MODE` are **read-only at runtime** (they are
   fixed by the BD config defaults: BASIC_ONLY / AND / ALWAYS).
4. **Trigger compare syntax.** Match a slice of the probe:
   ```tcl
   set pr [get_hw_probes -of_objects $ila -filter {NAME == {system_i/axi_hp_top_dbg_bus}}]
   # bit0 == 1, everything else don't-care (36 hex digits = 144 >= 142 bits):
   set_property TRIGGER_COMPARE_VALUE "eq142'h[string repeat X 35]1" $pr
   ```
   `eq` = equality, `X` = don't-care. Matching a low-nibble pattern like
   `...B` can simultaneously enforce `state==COMPUTE` and `dbg_first_q8=1`
   because `dbg_first_q8=1` already implies `state==COMPUTE` (see the script
   comment for the nibble math).
5. **Trigger position trades pre- vs post-trigger samples.** Depth 1024 with
   `TRIGGER_POSITION 150` gives ~150 pre-trigger (the Q5 tail) and ~874
   post-trigger (CLEAR_ACC + the full 512-cycle COMPUTE + READ_RES). Choose it
   so the interesting window is fully captured.
6. **Poll status, don't sleep blind.** `STATUS.CORE_STATUS`,
   `STATUS.CAPTURE_STATUS`, `STATUS.SAMPLE_COUNT` tell you trigger fired
   (SAMPLE_COUNT == 1024 && CORE_STATUS == IDLE) and capture finished. Add a
   timeout so a never-triggered run doesn't hang the session.
7. **CSV output quirk.** `write_hw_ila_data -csv_file ...` writes one HEX
   column per probe plus a `TRIGGER` flag column. ILA probes are allocated in
   64-bit lanes, so a 142-bit probe produces the 142-bit value in one column
   and a pile of `system_i/<const0>_N` / `<const1>_N` padding columns tied to
   constants - ignore them. The bus value is a big hex number; slice it with
   the bit-layout comment.

### 1.6 Analyzing the captured trace (`linux/boot/ila_q8_trace.csv`)

The CSV has ~1024 rows (one per clock cycle). For this bug:

1. Slice each row's `dbg_bus` hex into `{dbg_g, p2_valid, p2_partial0,
   acc_rw_rd, acc_clr_cnt, state, ...}` using the bit layout.
2. Walk the first COMPUTE: confirm CLEAR_ACC touched every acc group, then
   watch the RMW accumulate per column.
3. **The decisive observation:** acc groups g=6,7 produced `p2_partial0 == 0`
   for **all 64 columns** (no dequant product ever accumulated -> scale read
   as 0), while g=0-5 were non-zero. Combined with the on-board `--compare`
   RAWDIFF rows (16-63 wrong, g6-7 zero), this localized the fault to
   **scale delivery into smem**, NOT acc-clear and NOT weight load.

That narrowed the search to the LOAD_SCALES path, which led (via the
functional-sim investigation) to the stale-`sc_burst_done` root cause: the
Q5 path sets the flag on its last block's `rd_done_rise`, never consumes it,
and the next Q8's first `LOAD_SCALES_W` consumes it on entry, dropping one
scale burst (see AGENTS.md "Bug 3").

### 1.7 ILA lessons learned (summary)

- Probe **existing wires only**; never let the probe alter synthesis
  structure (the `dbg_acc_r0` rule). If a signal is synthesized away, probing
  it may resurrect it.
- Prefer a **semantic software trigger** (`dbg_first_q8`) over raw-value
  triggers; make it self-latching so a long-running design can't re-fire it.
- Probe **both sides at once**: FSM cause-side (state, col_group, res_addr,
  counters) AND core effect-side (pipeline state, acc RMW, partial product)
  in a single bus - you get one window, use it to see causality.
- **Budget the width up front**; 238 bits failed placement, 142 fit. Trim
  48-bit values to 32. You can always widen on the next build.
- Run the **iVerilog regression with the dbg bus in place** - the RTL change
  must not break functional sim.
- **One session, one capture**: arm -> trigger -> read must be one script;
  closing JTAG disarms; reprogram the PL before re-arming a latched trigger.
- The `.ltx` probe file and re-fused `BOOT.BIN` are mandatory before any
  script works.
- The ILA trace gives a **localizing clue**, not necessarily the root cause.
  Here it proved the fault was in scale delivery; the *mechanism* (stale flag
  skipping a burst) was pinned down by making the functional sim reproduce it.
- After diagnosis, **revert to the non-ILA bitstream** for the production
  build (the debug bus + ILA add logic/routing/BRAM). Keep the ILA build only
  as long as it earns its keep.

### 1.8 Copy-paste board recipe

```bash
# 1. Add dbg_* outputs + dbg_bus in RTL; instantiate ila_0 in build_bd.tcl.
# 2. Rebuild:  vivado.bat -mode batch -source vivado_integration/build_bd.tcl
#    -> linux/boot/system_wrapper.bit + system_wrapper_ila.ltx
# 3. Re-fuse:  bootgen.bat -image boot.bif -o BOOT.BIN -w   (in linux/boot/)
# 4. Deploy BOOT.BIN + system_wrapper_ila.ltx to the SD card, boot to shell.
# 5. Host (keeps running):
#      vivado.bat -mode tcl -source linux/scripts/arm_wait_read_ila_q8.tcl
# 6. Board (while step 5 is armed):
#      ./tmac model.tmac    (the --nowarmup flag used during the 2026-08-16 ILA
#      campaign was removed 2026-08-17 - single-run is now the default)
# 7. Step 5 polls, captures, and writes linux/boot/ila_q8_trace.csv.
# 8. Slice the CSV hex by the bit-layout comment and correlate.
```

---

## 2. iVerilog functional simulation debugging

The sim is the cheapest debug tool and the source of most of this project's
fixes - but only if it **faithfully reflects the RTL** and its testbench
**actually exercises the failing path**.

### 2.1 THE lesson of this session: testbench address coverage

The stale-`sc_burst_done` bug was **not** reproduced in sim for weeks because
`tb_hw_fsm_comprehensive.v`'s DDR model `ddr_mem` was only 4 MB
(`[0:524287]` -> addresses 0x00000000-0x003FFFFF), while Test 10 placed the
Q5 descriptor/weights/acts at 0x00400000-0x00412000 - **out of bounds**.
Every out-of-range read returned `X`, so the descriptor's `tensor_type` read
as `X` and **fell through to the Q8 default dispatch**: the Q5 path (which
sets the stale flag) never ran, and the test passed for the wrong reason.

Audit rule: **every address a test writes/reads must be inside the memory
model, and an in-bounds X should be a test failure, not a silent fall-through.**
Growing `ddr_mem` to 8 MB (`[0:1048575]`) instantly turned Test 10 from
"passes" into "fails on rows 16+" - the exact board pattern. If a board bug
never reproduces in sim, check the testbench's address coverage BEFORE
rethinking the hardware theory.

### 2.2 X-propagation discipline

`X` in a memory or descriptor is a double-edged sword:

- It can **mask** bugs (the Q5 type read as X -> Q8 default -> test never
  exercised the Q5 path).
- It can **expose** under-specified paths (an `X` cascade through a `case`
  default is usually a real dispatch hole).

When a sim "pass" disagrees with the board, ask: "did the test actually
execute the path I think it did?" Add a dispatch log or assert at test time.

### 2.3 Uniform data masks bugs

Three 2026-08-12 Q8 bugs (address advancement stuck at 0, column-major vs
row-group-major weight layout, tiny un-normalized scales) were latent for
months because every test used identical weights/scales/activations - any
address reads the same value, any layout gives the same result, any scale
size gives the same zero. Rule: **always add non-uniform patterns** (distinct
per row / per col / per scale / per group). See `tb_hw_fsm_comprehensive.v`
Test 8/9 non-uniform additions and `test_fpga_cores.cpp` row/col patterns.

### 2.4 Sim-only constructs

- `__ICARUS__`-gated `initial` blocks that zero memories give sim a
  deterministic starting point that **synthesis ignores**. Do not confuse
  "sim is clean" with "hardware is initialized". (The 2026-08-16 `INIT=0`
  LUTRAM change was exactly this trap: it made the sim-and-board consistent
  but was the wrong theory - the bug was the FSM stale flag.)
- If sim and hardware differ, the discrepancy is often a sim-only shortcut.
  Find it (the `__ICARUS__` gate, an idealized FP32 model, a too-small memory
  model, an over-perfect AXI model).

### 2.5 The funnel principle (make the sim match the RTL)

`sim/tmac_gguf.cpp` was re-wired to call the **bit-exact golden models**
(`golden::q5_tile_golden`, `golden::q8_tile_golden` in `sim/golden_model.hpp`)
instead of idealized FP32 matmuls. Once the sim reflects the real fixed-point
RTL, bugs that only appear with real data (Q5 negative-d sign flip, S48
truncation) surface in sim first. A sim that "works" with idealized math
proves nothing about the RTL.

### 2.6 Reproduce-first discipline

Never validate a fix against a test that never exercised the bug. The order
matters:

1. Make the sim **reproduce** the failure (grow `ddr_mem` -> Test 10 FAILs).
2. Apply the fix (`sc_burst_done <= 0;` in the Q8 dispatch).
3. Test 10 PASSes; run the whole regression (both suites, 10/10).
4. Revert the fix mentally: without it, Test 10 must FAIL again. If it does,
   the test is a faithful canary for the bug.

### 2.7 Using `$display` debug instrumentation

Temporary `$display` in the RTL is a fast way to trace a sim:
- Print FSM state transitions, dispatch decisions, and burst-done flags at
  the exact points of interest.
- It works because the tb drives the RTL; the prints tell you what the RTL
  actually did (e.g. "dispatch Q5_0", "Q5_READ_RES sc_burst_done=1",
  "FETCH_DESC done type=0xXX").
- **Remove them after the root cause is found** and the fix is verified - the
  RTL must ship clean. This session added ~8 prints and removed all of them.
- If a print fires with an unexpected value (e.g. `type=0xXX`), that print
  just found the tb/model bug - follow it.

### 2.8 Verilog/iVerilog gotchas encountered

- **Task `input integer` truncates 64-bit words.** `tb_matmul_q8.v`'s `wr`
  task declared `din` as `integer` (32-bit signed), truncating the upper 32
  bits of a 64-bit weight word - banks 4-7 got 0x00. Use `input [63:0]`.
- **`break` is unsupported** in iVerilog wait loops - use a `poll_count` flag.
- **`@*` array sensitivity warnings** for `acc_b0..7` - expected, cosmetic.
- **Single-flight read master**: re-`rd_start` while busy is dropped. This
  is why the stale-flag consume (which re-issues LOAD_SCALES for a burst
  already in flight) silently loses a burst. When tracing a DMA bug, always
  ask what the master does with a second start.

### 2.9 Waveform debugging

`$dumpfile`/`$dumpvars` + GTKWave for RTL micro-sequencing. The descriptor
chains are long; keep the window tight and use the `$display` state prints to
find the cycle to zoom into.

---

## 3. Other debugging means

### 3.1 On-chip register debug map

The FSM exposes live debug registers readable over AXI4-Lite from XSDB or
Linux `/dev/mem` (`REG_DEBUG` 0x28, `REG_Q8_DEBUG` 0x3C, `REG_Q5_DBG_*`
0x40-0x68 - see AGENTS.md). Strengths: no rebuild needed, coarse state/hang
detection, timeout source. Weakness: a frozen snapshot - you cannot see
causality across cycles (that is exactly when you reach for the ILA). Use
`REG_CLK_CNT` (0x2C) to measure timing, and the timeout latch
(`TIMEOUT_ERROR` state, `timeout_src`) to find which burst hung.

### 3.2 On-board A/B compare flags in `tmac`

- `--selftest`: minimal CPU_OP DDR copy - isolates the PL DDR path from all
  compute.
- `--compare`: run CPU+FPGA per matmul and print per-tile maxdiff +
  RAWDIFF. The workhorse for quantized-path diffs.
- `--trace`: per-layer hidden-state comparison vs the sim's ground truth -
  used to verify the Q6_K dequant fix (layer norms L0..L23 all match).

### 3.3 Bit-exact golden models (`sim/golden_model.hpp`)

Fixed-point C++ models of the Q8/Q5 cores, shared by:
- the C++ sim (`sim/tmac_gguf.cpp` - the funnel),
- the bare-metal tests (`test_fpga_cores.cpp` `test_q5_golden`),
- the Linux `tmac` (`gold=` vs `acc=` per-tile).

They let you prove "the core arithmetic is bit-exact; the error is in the
host scaling / precision" (Q5 d_pre S16 bottleneck) or the reverse, and pin
down whether a mismatch is RTL or host.

### 3.4 Host-side C reproductions (no hardware)

- `q6k_cmp.c`: byte-compare Linux vs sim Q6_K/Q4_K dequant over all 3.5M
  samples on x86 - found the missing `+bo` block offset at exactly idx=256.
- `linux_repro.c`: verbatim Linux forward-pass on x86 - reproduced the
  layer-0 115M norm explosion and proved the fix dropped it to 17.49.
- These cost minutes and need no board; run them whenever the suspect is a
  host-side numeric path shared with the sim.

### 3.5 Bare-metal hardware test programs

`test_fpga_cores.elf` (13 tests) drives the FPGA cores directly from the ARM
with no OS and prints PASS/FAIL + raw accumulator words - the fastest way to
verify a specific core behavior on silicon (bit-exact golden, 14-group
row-pattern, negative acts, etc.). Loaded via XSDB
(`run_test_fpga_cores.tcl`).

### 3.6 XSDB / DAP / ps7_init

See `docs/debug_procedures.md` - a complete reference for JTAG sessions,
DAP error recovery (power-cycle discipline), ps7_init hang recovery, FCLK
enable, AFI register debugging, and FSM-hang decoding via `REG_DEBUG`.

---

## 4. General debugging methodology (the funnel)

1. **Start broad, narrow fast.** Compare end-to-end output, then per-layer,
   per-tensor, per-tile, per-group, per-cycle. Each step needs a measurement
   (`--compare`, `--trace`, per-group isolation `num_groups=1..14`, ILA).
2. **Sim and board must agree.** When they don't, one of them is wrong -
   find which before building new theories. The 4 MB `ddr_mem` case was a
   silent sim error that masqueraded as "board-only bug".
3. **Reproduce before you fix.** A fix validated against a test that never
   exercised the bug is worthless (the stale-flag lesson).
4. **Non-uniform data always.** Uniform data hides address, layout, scale,
   and shift bugs.
5. **Isolate the layer of the fault.** Is it data (host preprocess), golden
   (host model), core arithmetic (RTL), or handshake (FSM/masters)? The Q8
   first-run bug survived exactly this triage: data/golden/core all clean,
   FSM handshake guilty.
6. **Know each tool's blind spot.** Registers: no causality. Sim: only as
   good as its models + address coverage. ILA: only what you probed, one
   window at a time, and it can change what it observes if a probe revives
   dead logic.
7. **When you find a bug, fix the tool that hid it too.** Growing `ddr_mem`
   and the non-uniform tests are permanent regressions that will catch the
   next instance of this bug class.