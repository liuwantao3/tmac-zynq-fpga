# Test Infrastructure

## Overview

Board-level testing uses **XSDB** (Xilinx System Debugger) Tcl scripts. All scripts live in `vivado_integration/sw/`. They follow a common flow:

```
power-cycle → connect → fpga -file → source ps7_init.tcl → ps7_*_init → AFI config → test → stop
```

**Key constraint:** XSDB is the only tool that provides `mwr`/`mrd`/`mask_write` commands needed for PS7 register config and DDR access. Vivado TCL (`vivado -mode tcl`) can program the FPGA via `program_hw_devices` but lacks raw memory access commands.

---

## AXI4-Lite Register Map (hp_fsm_top)

Base address `0x43C00000`. These registers control the descriptor-chain FSM (used by `run_hp_fsm_test.tcl`, `test_multiburst.tcl`).

| Offset | Name | Type | Description |
|--------|------|------|-------------|
| 0x00 | `reg_start` | W | Bit[0] = start chain (auto-clears on state != IDLE) |
| 0x04–0x10 | — | — | (reserved, may be 0) |
| 0x14 | `reg_status` | R | `[15]=busy`, `[9]=wr_done`, `[8]=rd_done` |
| 0x18 | `reg_desc_base` | R/W | Descriptor base address (DDR physical) |
| 0x1C | `reg_desc_tail` | R/W | Tail index (number of descriptors) |
| 0x20 | `reg_desc_head` | R | Head index (read-only, auto-advances) |
| 0x24 | — | — | (reserved) |
| 0x28 | `reg_debug` | R | See DEBUG register bitfield below |
| 0x2C | `reg_clk_cnt` | R | Free-running 100 MHz clock counter |
| 0x30 | `reg_clk_cnt_slow` | R | Clock counter ÷ 1024 |

### DEBUG register (0x28) bitfield

| Bits | Field | Description |
|------|-------|-------------|
| 31:28 | `state` | FSM state encoding (see below) |
| 27 | `rd_done` | Read burst done pulse |
| 26 | `wr_done` | Write burst done pulse |
| 25 | `rd_busy` | Read master busy |
| 24 | `wr_busy` | Write master busy |
| 23:16 | `act_remaining[7:0]` | Low byte of remaining bytes to read |
| 15:8 | `rd_len` | Number of AXI beats - 1 in current burst |
| 7:0 | `act_byte_idx` | Current byte index within current burst |

### FSM state encoding

| Value | State | Description |
|-------|-------|-------------|
| 0 | `IDLE` | Waiting for `reg_start` |
| 1 | `FETCH_DESC` | Reading descriptor from DDR |
| 2 | `FETCH_DESC_W` | Draining descriptor read data |
| 3 | `LOAD_ACT` | Starting HP read burst for activation data |
| 4 | `LOAD_ACT_W` | Draining HP read data into act_buf |
| 5 | `WRITE_RES` | Starting HP write burst for result |
| 6 | `WRITE_RES_W` | Draining HP write of result data |
| 7 | `DONE` | Chain complete (cleared by next `reg_start`) |

---

## AXI4-Lite Register Map (hp_loopback_top)

Base address `0x43C00000`. Used by `run_hp_loopback.tcl`.

| Offset | Name | Description |
|--------|------|-------------|
| 0x00 | `CTRL` | Bit[0] = start |
| 0x04 | `RD_ADDR` | DDR source address for HP read |
| 0x08 | `RD_LEN` | Burst length in beats minus 1 (max 15 for AXI3) |
| 0x0C | `WR_ADDR` | DDR destination address for HP write |
| 0x10 | `WR_CNT` | Number of 64-bit words to write |
| 0x14 | `STATUS` | `[9]=wr_done`, `[8]=rd_done` |
| 0x20..0x5C | `DBG_W0_lo`..`DBG_W7_hi` | 16 × 32-bit read buffer dump registers |

---

## Script Catalog

### Combined-Flow Scripts (Single Session)

These scripts do it all: `fpga -file` → `ps7_init` → AFI → test → verify → cleanup.

| Script | Lines | Purpose | Status |
|--------|-------|---------|--------|
| `run_hp_loopback.tcl` | 283 | HP read from DDR-A, write to DDR-B, verify | ✅ PASS on hw (2026-06-20) |
| `run_hp_fsm_test.tcl` | 309 | FSM descriptor chain: fetch → act load → write result | ✅ PASS on hw (60B) |
| `run_hp_write_test.tcl` | 88 | HP write standalone (OCM AFI) | ✅ PASS on hw |
| `run_hp_read_test.tcl` | 181 | HP read standalone (capture buffers) | ✅ PASS on hw |
| `test_multiburst.tcl` | 134 | Multi-burst HP reads via FSM (temp dir) | ⚠️ 60B pass, 65B+ rd_len bug |

### Setup & Diagnostics

| Script | Purpose |
|--------|---------|
| `probe_targets.tcl` | Enumerate JTAG targets, test `mrd` access on each |
| `debug_targets.tcl` | Show targets before/after `fpga -file` |
| `recover_board.tcl` | `connect → rst -dap → reconnect` |
| `clear_dap.tcl` | Write DAP ABORT + CTRL/STAT registers to clear sticky errors |
| `recover_board.tcl` | `rst -dap` to clear DAP state |
| `diag_ddr.tcl` | Readback DDR at known addresses |
| `diag_deep.tcl` | Deep register diagnostic dump |
| `diag_afi.tcl` | AFI register readback |
| `quick_afi_test.tcl` | Minimal AFI read check |
| `check_segments.tcl` | Address segment verification |
| `check_dap_cmds.tcl` | Probe DAP capabilities |

### Vitis & Build

| Script | Purpose |
|--------|---------|
| `build.tcl` | Build Vivado project from Tcl |
| `rebuild.tcl` | XSCT: rebuild Vitis app |
| `run_elf.tcl` | XSDB: program + init + run ELF |
| `check_hsi.tcl` | HSI (Vitis) environment check |
| `check_xsa.tcl` | XSA file validation |
| `gen_ps7_init.tcl` | Generate ps7_init from XSA |

---

## Standardized Test Flow (hp_fsm_top)

This is the proven flow from `test_multiburst.tcl`:

```tcl
# === Setup ===
configparams force-mem-accesses 1
connect; after 5000
targets -set -filter {name =~ "*Cortex-A9*#0*"}; after 200

# === 1. Program FPGA ===
fpga -file {vivado_integration/proj_bd/matmul_bd.runs/impl_1/system_wrapper.bit}; after 1000

# === 2. Initialize PS7 ===
source {vivado_integration/proj_bd/matmul_bd.gen/sources_1/bd/system/ip/system_ps7_0/ps7_init.tcl}
ps7_mio_init_data_3_0; ps7_pll_init_data_3_0; ps7_clock_init_data_3_0
ps7_ddr_init_data_3_0; ps7_peripherals_init_data_3_0; after 300
ps7_post_config_3_0; after 200
if {([read32 0xF800010C] & 7) != 7} { puts "PLL NOT LOCKED"; exit 1 }

# === 3. Configure AFI0 ===
slcr_unlock; mwr -force 0xF8000910 0x0000000F; slcr_lock
# NOTE: AFI0_FIFO_PARTITION write may be ignored by boot ROM (write-once).
# The real workaround for 0-block write FIFO is single-beat AXI writes (AWLEN=0)
# in axihp_write_master.v — see docs/debug_log.md for details.
mwr -force 0xF8008004 0x00000044; after 20   # PART = 4R+4W (may be ignored)
mwr -force 0xF8008000 0x00000005; after 20   # CTRL = enable + light-weight
mwr -force 0xF8008008 0x00000001; after 20   # WRCHAN enable
mwr -force 0xF800800C 0x00000001; after 20   # RDCHAN enable

# === 4. Verify GP0 (AXI4-Lite) access ===
set clk_cnt [read32 0x43C0002C]
if {$clk_cnt == 0} { puts "WARN: clock counter 0 — PL not clocked!" }

# === 5. Write descriptor to DDR ===
mwr -force <DESC_ADDR+0x00> <next_addr>
mwr -force <DESC_ADDR+0x04> <weight_addr>
mwr -force <DESC_ADDR+0x08> <act_addr>
mwr -force <DESC_ADDR+0x0C> <result_addr>
mwr -force <DESC_ADDR+0x10> 0
mwr -force <DESC_ADDR+0x14> 0
mwr -force <DESC_ADDR+0x18> <act_total_bytes>
mwr -force <DESC_ADDR+0x1C> 0

# === 6. Configure and start FSM ===
mwr -force 0x43C00018 <DESC_ADDR>   # reg_desc_base
mwr -force 0x43C0001C 1             # reg_desc_tail
mwr -force 0x43C00000 1             # reg_start

# === 7. Poll for done ===
set done 0
for {set i 0} {$i < 200} {incr i} {
    set s [read32 0x43C00014]       # reg_status
    if {[expr {$s & 0x8000}] == 0} { set done 1; break }
    after 50
}

# === 8. Verify results ===
set debug [read32 0x43C00028]       # reg_debug
set status [read32 0x43C00014]      # reg_status
# Read RES_ADDR + offset for verification

# === 9. Cleanup ===
catch {stop}; after 200
```

---

## Descriptor Format (32 bytes, DDR)

| Offset | Field | Size | Description |
|--------|-------|------|-------------|
| 0x00 | `next_addr` | 4 | Next descriptor address (0 = end) |
| 0x04 | `weight_addr` | 4 | Weight data address (not used by hp_fsm_top loopback) |
| 0x08 | `act_addr` | 4 | Activation data address |
| 0x0C | `result_addr` | 4 | Result write address |
| 0x10 | — | 4 | (reserved) |
| 0x14 | — | 4 | (reserved) |
| 0x18 | `act_total_bytes` | 4 | Total bytes to read (little-endian) |
| 0x1C | — | 4 | (reserved, must be 0) |

---

## HP0 Write FIFO = 0 Blocks (Workaround)

### Problem
Boot ROM sets `AFI0_FIFO_PARTITION=0x00000007` → write channel = 0 FIFO entries. Write-once after reset.

### Why This Matters
Multi-beat AXI write bursts (AWLEN > 0) require the interconnect to buffer beats. With 0 FIFO entries, burst data is silently dropped even though AXI handshakes complete (speculative AWREADY/WREADY/BVALID).

### Workaround: Single-Beat Writes
The `axihp_write_master.v` uses **single-beat AXI writes** (AWLEN=0, AWSIZE=2 → 4 bytes). Each 64-bit word from the FSM is split into two 32-bit single-beat transactions. The interconnect sinks single-beat writes directly to the DDR controller without FIFO buffering.

This is transparent to the FSM — it writes one 64-bit word per `wready` handshake, and the write master handles the 2×32-bit split internally. The `wr_count` at `hp_fsm_top.v:340` = `act_total_bytes / 8` counts 64-bit words.

### Verification
Hardware HP loopback test (2026-06-20) confirmed all write transactions complete with correct data. All subsequent FSM chain tests use this write path.

### Why Not ACP
ACP has no FIFO limitation and provides cache coherence, but switching would require block design changes, AXI ID routing, and risks L1/L2 cache pollution. HP0 single-beat writes are sufficient for result writeback.

### References
- `verilog/axihp_write_master.v` — single-beat write implementation
- `docs/debug_log.md` — full root cause analysis and workaround verification history

## AXI3 Burst Limit (Critical!)

PS7 HP ports are **AXI3**, not AXI4. Max burst length = **16 beats**. With ARSIZE=2 (32-bit reads):

- **Max per burst**: 16 × 4 = **64 bytes**
- **ARLEN max**: 15 (0-indexed: 0..15 = 16 beats)
- **For transfers >64 bytes**: The FSM must split into multiple bursts

The `hp_fsm_top.v` LOAD_ACT/LOAD_ACT_W states implement multi-burst: each burst is capped at 64 bytes, and `act_remaining` tracks leftover bytes for subsequent bursts.

**Bug fixed (2026-06-21):** `rd_len` computation underflows when `act_remaining < 4`: `(1 >> 2) - 1 = 0xFF` → ARLEN=255 (rejected by PS7). Fixed by adding `act_remaining >= 4` guard with `rd_len <= 8'd0` fallback for small remainders.

---

## XSDB Target Visibility

The Digilent HS-2 cable detection is intermittent. Recovery steps:

1. **Power-cycle** (disconnect ALL USB from board, wait 60s, reconnect JTAG only)
2. **`fpga -file`** — programming the PL often makes A9 targets visible
3. **Target index scan** — iterate all targets by index after `fpga -file`
4. **`targets -set -filter {name =~ "*Cortex-A9*#0*"}`** with `catch` to ignore failures
5. **Vivado HW Manager fallback** — `vivado -mode tcl` with `open_hw_manager` can see targets when XSDB cannot

If `targets` shows 0 devices (no DAP, no xc7z010):
- Check USB: `Get-PnpDevice | Where-Object {$_.InstanceId -match "0403"}` should show FTDI chip
- Kill stale `hw_server.exe` and `cs_server.exe` processes
- Retry with longer `after` delay (10-15s)

---

## Power-Cycle Protocol

Required before every test session. The PS7 PLLs cannot be re-initialized if already locked.

1. Disconnect ALL cables from board (JTAG USB, any barrel jack power)
2. Wait **60+ seconds** for capacitors to drain
3. Reconnect JTAG USB only
4. Wait for board LEDs to stabilize (~5s)
5. **Verify**: `mrd 0xF800010C 1` should show PLL_STATUS=0 (all unlocked) before ps7_init

**Do not** use `rst -processor` between test runs — it corrupts DAP irrecoverably until next power-cycle. Use `stop` at end of test to halt ARM cleanly.
