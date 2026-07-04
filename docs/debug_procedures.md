# Debug Procedures

## XSDB Quick Reference

### Connect & Target Selection

```tcl
configparams force-mem-accesses 1
connect; after 5000
targets -set -filter {name =~ "*Cortex-A9*#0*"}; after 200
```

If target selection fails:
```tcl
targets                                      # list all targets
targets -set -filter {name =~ "*DAP*"}; after 200
rst -dap; after 2000                         # DAP reset
targets -set -filter {name =~ "*Cortex-A9*#0*"}; after 200
```

### Common XSDB Commands

```tcl
mwr -force <addr> <value> [count]     # memory write (32-bit)
mrd <addr> [count]                    # memory read
mask_write <addr> <mask> <value>      # masked write (read-modify-write)
slcr_unlock {} -> mwr -force 0xF8000008 0x0000DF0D
slcr_lock {}   -> mwr -force 0xF8000004 0x0000767B
stop                                   # halt ARM
con                                    # resume ARM
rst -processor                         # reset ARM (AVOID — corrupts DAP)
rst -dap                               # reset DAP (safe)
fpga -file <bitfile>                   # program PL
source <ps7_init.tcl>                  # load PS7 init procs
```

### proc read32

```tcl
proc read32 {addr} {
    set r [mrd $addr 1]
    if {[regexp {:\s+([0-9A-Fa-f]+)} $r full data]} { return [expr "0x$data"] }
    return -1
}
```

---

## DAP Error Recovery

### Error Codes

| Status | Meaning | Recovery |
|--------|---------|----------|
| `0x30000001` | AP Transaction Error (sticky) | Power-cycle. Usually transient after `fpga -file`. |
| `0x30000021` | STICKYERR + AP Transaction Error | Power-cycle required. PS7 power domain in bad state. |
| `0xF0000021` | AHB AP transaction error | Power-cycle. DAP connection corrupted by `rst -processor`. |
| `0x00000000` | DAP reads return 0 for all PS7 regs | PS7 AHB interconnect broken. Power-cycle. |

### Primary Recovery: Power-Cycle

Required when:
- DAP shows `0x30000021` or `0xF0000021`
- No A9 targets visible
- PS7 registers read as 0x00000000

**Must disconnect ALL cables** (not just JTAG). The Zynq PS7 has decoupling caps that hold charge for 30+ seconds. Wait 60+ seconds minimum.

### Secondary Recovery: DP Register Clear

Sometimes works for `0x30000001` (bit 0 only, not bit 5):

```tcl
targets -set -filter {name =~ "*DAP*"}
mwr -dp 0x4 0x00000000    # clear DP CTRL/STAT sticky bits
# or via Vivado TCL:
set_hw_reg_value $dap {PS7 DBG CTRL} 0x50000000
```

### Clean Shutdown Between Test Runs

Always halt ARM at the end of a test:
```tcl
targets -set -filter {name =~ "*Cortex-A9*#0*"}; after 50
stop; after 200
```
This prevents DAP corruption from free-running ARM during next session.

**Never use `rst -processor`** — it corrupts DAP irreversibly until power-cycle.

---

## ps7_init Hang Recovery

### Symptoms
- `ps7_pll_init_data_3_0` hangs at `mask_poll 0xF800010C`
- `ps7_ddr_init_data_3_0` hangs at `mask_poll 0xF8000B74`

### Root Cause
PLLs are already configured from a prior session. The reset+re-lock sequence in `ps7_pll_init_data_3_0` can't complete because PLLs are already locked.

### Fix
**Power-cycle the board.** A processor-only reset is insufficient.

### Workaround (when DDR is already initialized)
If the board was previously initialized and only the PL needs reconfiguring, skip ps7_init and just reconfigure AFI:

```tcl
slcr_unlock; mwr -force 0xF8000910 0x0000000F; slcr_lock
mwr -force 0xF8008004 0x00000044; after 20
mwr -force 0xF8008000 0x00000005; after 20
mwr -force 0xF8008008 0x00000001; after 20
mwr -force 0xF800800C 0x00000001; after 20
```

But verify PLLs are still locked first:
```tcl
if {([read32 0xF800010C] & 7) != 7} { puts "PLL NOT LOCKED — need power cycle" }
```

---

## FCLK_CLK0 Enable

FCLK_CLK0 is the 100 MHz PL clock from PS7. The SLCR bit `FPGA_CLK_CTRL[7]` (CLK0_EN) must be 1.

### DAP can't set CLK0_EN
Writing 0xF8000170 via DAP silently drops bit 7 (and bit 19 for CLK1). The SLCR write-protect bits block DAP access to clock enable bits.

### ARM code workaround
Load ARM boot code to OCM (0x00000000) that writes the SLCR registers directly:

```tcl
set code [list \
    0xE59FD038 0xE59F0038 0xE59F1038 0xE5801000 \  # unlock SLCR
    0xE59F0034 0xE59F1034 0xE5801000 0xF57FF04F \  # write FCLK enable
    0xE59F002C 0xE59F102C 0xE5801000 0xF57FF04F \  # another FCLK write
    0xE59F000C 0xE59F1020 0xE5801000 0xEAFFFFFE \  # spin
    0x0003FFF0 0x00100000 0xA5A5A5A5 \             # data: SP, marker
    0xF8000008 0x0000DF0D \                        # data: SLCR unlock
    0xF8000170 0x00480480 \                        # data: FCLK enabled value
    0x5A5A5A5A \                                    # data: marker
]
mwr -force 0x00000000 $code 24
targets -set -filter {name =~ "*Cortex-A9*#0*"}
stop; wrpc 0x00000000; con; after 500; stop
```

---

## HP Port Debug

### AFI0 Status Register (0xF8008010)

| Bit | Field | Meaning |
|-----|-------|---------|
| 0 | DONE | AFI initialization complete |
| 1 | W_ERR | Write error sticky |
| 2 | R_ERR | Read error sticky |
| 3 | B2B_ERR | Back-to-back error sticky |
| 4 | B2B_OVF | Back-to-back overflow sticky |
| 6 | UNSYNCHED | AFI not synced with PL |
| 7:4 | — | (reserved) |

Expected value after config: `0x1E` = DONE + all sticky bits set (normal).

### AFI0 Debug Register (0xF8008014)

Expected after HP operation: `0x0F` = all FIFO status OK.

### AFI0 Control (0xF8008000)

`AFI0_CTRL[7:6]` (64-bit enable) is **read-only** on Zynq-7010. The x16 DDR3 caps HP0 at 32-bit. Write attempts to bits[7:6] are silently ignored. Verified by write-verify loop.

### Key AFI Registers

| Address | Name | Value | Notes |
|---------|------|-------|-------|
| 0xF8008000 | AFI0_CTRL | 0x05 | enable + light-weight (bits[7:6] read-only = 0) |
| 0xF8008004 | AFI0_PART | 0x44 | **Boot ROM sets to 0x07** (write=0 blocks). Writing 0x44 is **ignored** — write-once after reset. Workaround: single-beat AXI writes (AWLEN=0) in `axihp_write_master.v` bypass FIFO. See `docs/debug_log.md`. |
| 0xF8008008 | AFI0_WRCHAN | 0x01 | write channel enable |
| 0xF800800C | AFI0_RDCHAN | 0x01 | read channel enable |
| 0xF8008010 | AFI0_STATUS | — | read-only |
| 0xF8008014 | AFI0_DEBUG | — | read-only |

---

## FSM State Debug via DEBUG Register

The DEBUG register (0x28) exposes live FSM state (RTL hp_fsm_top.v:793-804). When the FSM hangs, decode it:

```tcl
set dbg [read32 0x43C00028]
set state   [expr ($dbg >> 27) & 0x1F]     # [31:27] = 5-bit FSM state
set rd_done [expr ($dbg >> 26) & 1]        # [26] = rd_done
set wr_done [expr ($dbg >> 25) & 1]        # [25] = wr_done
set rd_busy [expr ($dbg >> 24) & 1]        # [24] = rd_busy
set wr_busy [expr ($dbg >> 23) & 1]        # [23] = wr_busy
set q8_busy [expr ($dbg >> 22) & 1]        # [22] = q8_busy
set q8_done [expr ($dbg >> 15) & 1]        # [15] = q8_done
set col_grp [expr ($dbg >> 11) & 0xF]      # [14:11] = col_group
set sc_idx  [expr $dbg & 0xFF]             # [7:0] = sc_byte_idx
puts "state=$state rd_done=$rd_done wr_done=$wr_done rd_busy=$rd_busy wr_busy=$wr_busy"
puts "q8_busy=$q8_busy q8_done=$q8_done col_group=$col_grp sc_byte_idx=$sc_idx"
```

### Hang State Interpretation (see docs/AGENTS.md for full 20-state table)

| State | Name | If hung, cause |
|-------|------|---------------|
| 0 | IDLE | FSM never started (check reg_start) |
| 1 | FETCH_DESC | HP read not completing (check AFI/DDR) |
| 3-4 | LOAD_ACT/_W | Act data not draining (rd_busy=1, rd_done=0 → AXI3 burst rejected) |
| 5-6 | WRITE_RES/_W | Write not completing (check wr_busy) |
| 7 | DONE | Chain incomplete (check start loopback) |
| 8-9 | LOAD_WEIGHT/_W | Weight HP read hung |
| 10-11 | LOAD_SCALES/_W | Scale HP read hung |
| 14 | COMPUTE_W | Q8 core timeout (check q8_done_rise) |
| 18 | TIMEOUT_ERROR | FSM hit timeout limit |
| All F's | — | PL not clocked |

---

## Test Session Checklist

### Before First Test (After Power-Cycle)

- [ ] `connect` succeeds without DAP errors
- [ ] `targets` shows A9 core(s) after `fpga -file`
- [ ] PLL status = 0x07 after ps7_init (all 3 PLLs locked)
- [ ] Clock counter at 0x43C0002C is non-zero (PL clocked)
- [ ] GP0 write-readback verify passes

### Between Tests (No Power-Cycle)

- [ ] `stop` ARM at end of previous test
- [ ] Skip `ps7_init` (PLLs already locked)
- [ ] Reconfigure AFI only
- [ ] If `fpga -file` reload needed, expect DAP warning but A9 should still appear

### When Things Break

1. **DAP 0x30000021**: Power-cycle (disconnect all cables, 60s)
2. **No A9 targets**: Try `fpga -file` then re-scan. If that fails, power-cycle.
3. **PS7 regs read 0**: Power-cycle (AHB interconnect broken)
4. **ps7_init hangs**: Already-configured PLLs. Power-cycle.
5. **Timeout polling STATUS**: Read DEBUG register, decode hang state, check AXI3 burst limit.
