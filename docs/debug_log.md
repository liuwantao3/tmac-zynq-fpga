# Debug Log: AXI HP Write Path & ACP Decision

## Root Cause: HP0/HP1 Write FIFO = 0 Blocks

### Discovery
During bare-metal testing of `axi_hp_int16_top.v` on MicroPhase Z7-Lite (xc7z010):
- AXI4-Lite control path (GP0) works — all register reads/writes correct
- HP read burst path works — weights and activations load correctly
- **HP write path fails silently** — write master completes AXI handshakes (AW/W/B) but data never reaches DDR

### Investigation
1. **PL-side observation**: Write master sees `AWREADY`, `WREADY`, and `BVALID` with `BRESP=OKAY`. But reading DDR via DAP shows old data (0xAAAAAAAA or 0xDEADDEAD, not written pattern).

2. **AFI FIFO Partition Register**: `AFI0_FIFO_PARTITION` at `0xF8008004` = `0x00000007`
   - Bits [3:0] = 0x7 → read channel = 7 FIFO entries
   - Bits [11:8] = 0x0 → **write channel = 0 FIFO entries**
   - Total FIFO depth = 8 entries, ALL allocated to reads, ZERO to writes

3. **Write-once lock**: The FIFO partition register is write-once after reset. Boot ROM writes it during DDR init. ARM writes with SLCR unlock are **silently ignored**.

4. **Speculative AXI response**: The AXI interconnect (`axi_hp`) provides speculative `AWREADY + WREADY + BVALID` responses even though the write data never enters the PS7's 0-block write FIFO. This gives the false appearance of success.

5. **SLCR unlock verified working**: MIO pin register test (`0xF8000240`) confirmed 0xDEADBEEF writes correctly — SLCR unlock is functional.

### Attempted Fixes (all failed)

| Fix | Result |
|-----|--------|
| Write `AFI0_WR_CHANNEL_CTRL=1` (`0xF8008008`) | Writable, but useless with 0-block FIFO |
| Write `AFI0_FIFO_PARTITION=0x707` (`0xF8008004`) | **Ignored** — write-once after boot |
| Enable bypass mode `AFI0_CTRL[1]=1` | Does not overcome 0-block FIFO |
| **Switch to HP1** (`AFI1_CTRL=0x03`, `AFI1_WR_CHANNEL_CTRL=0x01`) | Same boot ROM: `AFI1_FIFO_PARTITION=0x00000007` — **identical 0-block write FIFO** |
| Try writing `AFI1_RD_CHANNEL_CTRL` (`0xF800900C`) | Not writable — same as AFI0_RD_CHANNEL_CTRL |

### HP1 Register Dump (ARM read, verified by AFI experiment)
```
AFI1_CTRL            (0xF8009000) = 0x00000003  (enabled + bypass)
AFI1_FIFO_PARTITION  (0xF8009004) = 0x00000007  (read=7, write=0)
AFI1_WR_CHANNEL_CTRL (0xF8009008) = 0x00000001  (enabled)
AFI1_RD_CHANNEL_CTRL (0xF800900C) = 0x00000000  (disabled, not writable)
```

**Conclusion**: Both HP0 and HP1 have 0-block write FIFOs assigned by boot ROM. No software path can change this. The HP ports are read-only for all practical purposes.

### AFI1 Connectivity Fix
`ps7_post_config_3_0` in `ps7_init.tcl` was modified to enable AFI1:
```tcl
mwr -force 0xF8009000 0x00000003  ;# AFI1 enable + bypass
mwr -force 0xF8009008 0x00000001  ;# AFI1 write channel enable
mwr -force 0xF800900C 0x00000001  ;# AFI1 read channel enable
```
This resolved the "reading AFI1 registers hangs ARM" issue from earlier debug sessions.

---

## Cache Coherence Discovery

**Problem**: ARM writes to DDR were not visible to DAP reads, and vice versa.

**Root cause**: Default MMU map marks DDR as cacheable (write-back). ARM's `volatile` writes go to L1 cache + PL310 L2 cache. Without proper cache maintenance:
- DAP reads DDR directly (bypasses cache) → sees stale data
- PL's HP reads go directly to DDR → sees stale data after CPU writes

### Initial Fix Attempt
`Xil_DCacheFlushRange()` and raw DCCMVAC (`MCR p15, 0, ..., c7, c10, 1`) only flush L1 D-cache. L2 (PL310) stays dirty.

### Working Fix: PL310 L2 Cache Maintenance
```c
// PL310 at 0xF8F02000 (Zynq-7010)
#define L2C_CLEAN_WAY  (0xF8F02000 + 0x7B8)  // Clean all 8 ways → write data to DDR
#define L2C_CLEAN_INV  (0xF8F02000 + 0x7FC)  // Clean + invalidate all 8 ways
#define L2C_SYNC       (0xF8F02000 + 0x730)  // Poll bit 0 until 0

void flush_to_ddr(void) {
    // 1. Flush L1 D-cache (DCCMVAC for each cache line)
    for (addr = start & ~31; addr < end; addr += 32)
        __asm__("mcr p15, 0, %0, c7, c10, 1" : : "r" (addr));
    __asm__("dsb");
    // 2. Clean entire L2 cache by way
    *(volatile unsigned*)L2C_CLEAN_WAY = 0xFF;
    __asm__("dsb");
    while (*(volatile unsigned*)L2C_SYNC & 1);  // wait for completion
    __asm__("dsb"); __asm__("isb");
}
```

This fix was added to `hp_test.c` but its effectiveness was not yet verified by hardware test (spin 2 failed).

---

## Verilog Bug Fixes (axi_hp_int16_top.v)

### 1. Stray Byte Consumption (RD_WT/RD_ACT)
**Symptoms**: Last activation value corrupted, computation results wrong.
**Root cause**: After HP read burst completes, `data_valid` stays high for 1 cycle with `hp_read_busy` already low. Data consumption logic consumed this extra byte.
**Fix**: Added `&& hp_read_busy` guard to data consumption in RD_WT and RD_ACT states.

### 2. DRAIN 6-bit Counter Wrap
**Symptoms**: FSM never exits DRAIN state.
**Root cause**: `reg [5:0] idx` wraps at 63. Condition `idx < 64` is **always true** for 6-bit counter.
**Fix**: Widened to `reg [6:0] idx` (7 bits, range 0-127).

### 3. FSM Auto-Restart
**Symptoms**: After first computation completes, FSM immediately restarts without CPU trigger.
**Root cause**: `reg_ap_ctrl[0]` (start bit) never cleared by AXI4-Lite write handler.
**Fix**: `if (state != IDLE) reg_ap_ctrl[0] <= 0;` auto-clears start bit once FSM leaves IDLE.

### 4. Write Master `done` Stuck High
**Symptoms**: After first write burst, WR_RES exits immediately on next operation without writing anything.
**Root cause**: Write master's `done` output stays 1 after completion.
**Fix**: Use `hp_write_busy_fall` (falling edge on busy) instead of `!hp_write_busy && hp_write_done`.

### 5. HP Read Master Byte Duplication (ROOT CAUSE of wrong results)
**Symptoms**: In simulation, all 64 results = 2080 (golden) after fix. Before fix, random wrong values.
**Root cause**: `data_out` always showed `rdata_buf[buf_idx][7:0]`. Since `buf_idx` increments via NBA, the same byte was presented twice — once with old buf_idx, once with new. Cross-beat: last byte of beat consumed again during WAIT_R/IDLE transition.
**Fix 1**: On non-last byte consumption, immediately advance `data_out` to next byte.
**Fix 2**: On last byte consumption, clear `data_valid <= 0`.

### iVerilog Verification
After all 5 fixes: `tb_hp_verify.v` passes — all 64 results = 2080 (golden value).

---

## Debug Tools & Workflow

### Batch Build Pipeline
```bash
# Step 1: Vivado block design + bitstream
C:\Xilinx\Vivado\2023.1\bin\vivado.bat -mode batch -source vivado_integration\build_bd.tcl

# Step 2: Rebuild Vitis app
C:\Xilinx\Vitis\2023.1\bin\xsct.bat vivado_integration\sw\rebuild.tcl

# Step 3: Single XSDB session
C:\Xilinx\Vivado\2023.1\bin\xsdb.bat vivado_integration\sw\run_hp.tcl
```

### DAP / JTAG Lessons

| Lesson | Detail |
|--------|--------|
| DAP sticky after `fpga -file` | Every `fpga -file` sets DAP error (0x30000021). Only full power cycle clears it. |
| `rst -system` does NOT clear DAP | Despite the name, system reset doesn't clear DAP sticky errors. |
| PS7 init must run on APU/DAP | `targets -set 1` works; xc7z010 target returns "Context does not support memory read" |
| `mrd` cannot read PL registers | Addresses 0x43C00000+ return "Blocked address". Use ARM core reads. |
| `mrd` output not on stdout | Goes to TCF channel. Use `puts [mrd ...]` to capture in script. |
| Target filtering fails with status | `filter {name =~ "DAP"}` fails because target name includes "(AHB AP transaction error...)". Use `targets -set 1` by index. |
| `stop` fails after `con` | "Cannot halt processor core, timeout" — core cannot be stopped once running. |

### OCM as Cache-Free Diagnostic Area
OCM (`0x00000000` range) is non-cacheable by default. Spin markers in OCM allow DAP to detect ARM progress without cache concerns. Used in `hp_test.c`:
```c
*(volatile unsigned*)(OCM_BASE + offset) = 1;  // ARM writes marker
while (*(volatile unsigned*)(OCM_BASE + offset));  // ARM spins
// DAP reads OCM marker directly (no cache involved)
```

---

## Decision: Switch from HP to ACP

### Why HP is Dead for Writes
- AFI0_FIFO_PARTITION and AFI1_FIFO_PARTITION both = 0x00000007 (write=0 blocks)
- Boot ROM locks these registers write-once after reset
- No software workaround (bypass mode, channel enable, etc.) overcomes 0-block write FIFO
- AXI interconnect provides speculative completion responses — data silently dropped

### ACP Alternative
The **Accelerator Coherency Port (ACP)** on Zynq-7010:
- 64-bit AXI3 slave port on PS7 (same protocol as HP)
- Connects directly to SCU (Snoop Control Unit) — **no AFI FIFO**
- No FIFO partitioning — reads and writes both work
- **Cache coherent** — automatic L1/L2 snooping. ARM doesn't need to flush cache before PL reads, and PL writes are immediately visible to ARM.
- Same burst capability: up to 16-beat, 64-bit bursts

### Comparison

| Aspect | HP0/HP1 | ACP |
|--------|---------|-----|
| Write path | **Dead** (0-block FIFO) | **Works** (no FIFO) |
| Read throughput | 64-bit burst, good | 64-bit burst, same |
| Cache coherence | Manual L1+L2 flush | Automatic via SCU snooping |
| FIFO partition | 8 total, all to reads | None needed |
| Protocol | AXI3 | AXI3 |
| ARID/AWID | Don't care | Must set for coherency level |

### ACP ID Bits
ACP uses ARID[3:0] / AWID[3:0] to control coherency:
- ID[0] = 1: Allocate in L2 cache on read
- ID[1] = 1: Allocate in L1 cache on read (pollutes cache for weight data)
- ID[2] = 1: Write snoops L1 cache
- ID[3] = 1: Write-back L2 cache

Recommended: ID = 0x0 for weights (non-cacheable reads), ID = 0xF for activations/results (fully coherent).

### Implementation Plan (Next Phase)
1. Modify `build_bd.tcl`: Enable `PCW_USE_S_AXI_ACP = 1`, route `axi_hp_top/M_AXI_HP → ps7/S_AXI_ACP`
2. Modify `axihp_read_master.v`: Drive ARID[3:0] with configurable ID value
3. Modify `axihp_write_master.v`: Drive AWID[3:0] with configurable ID value
4. Modify `axi_hp_int16_top.v`: Add ID registers, drive ID bits on read/write masters
5. Remove HP0/HP1 from `build_bd.tcl` (or keep as read-only fallback)
6. Update `ps7_init.tcl`: Remove AFI1 enable (no longer needed), ensure ACP clock enabled
7. Verify: Simulate ACP read/write in iVerilog, then hardware test

---

## Debug Session History (2026-06-12)

### Session 1: AFI Experiment + Spin 1
- FPGA programmed, PS7 initialized, ELF loaded
- Spin 1 completed: AFI experiment data captured (dbg[0..15] at DIAG_BASE+0x2D0)
- Key findings: AFI1 accessible, both AFI0+AFI1 FIFO partition = 0x00000007 (write=0)
- MIO write test (0xDEADBEEF) confirmed SLCR unlock works
- AFI1_RD_CHANNEL_CTRL = 0x00, not writable to 1

### Session 1: Spin 2 Failure
- After con, `targets -set -filter {name =~ "ARM*"}` failed — target name includes parenthesized status
- DAP memory read failed while ARM core running
- Script fixed: use `targets -set 1` (DAP by index) for all mrd calls
- PL310 L2 cache maintenance functions added (unverified)

### Vivado Build Times
- Full build (synthesis + implementation + bitstream): ~12 minutes
- Fast rebuild (incremental): ~3 minutes
