# Test Infrastructure

## Overview

Board-level testing uses **XSDB** (Xilinx System Debugger) Tcl scripts. All scripts live in `vivado_integration/sw/`. They follow a common flow:

```
power-cycle → connect → fpga -file → source ps7_init.tcl → ps7_*_init → AFI config → test → stop
```

**Key constraint:** XSDB is the only tool that provides `mwr`/`mrd`/`mask_write` commands needed for PS7 register config and DDR access.

---

## AXI4-Lite Register Map (hp_fsm_top)

Base address `0x43C00000`. See `docs/AGENTS.md` for complete register map and bitfields.
See `vivado_integration/sw/regs.h` for C header definitions.

| Offset | Name | Type | Description |
|--------|------|------|-------------|
| 0x00 | `reg_start` | W | Bit[0] = start chain (auto-clears) |
| 0x10 | `reg_q8_num_groups` | R/W | [3:0] column groups (0=single) |
| 0x14 | `reg_status` | R | `[15]=busy`, `[9]=wr_done`, `[8]=rd_done` |
| 0x18 | `reg_desc_base` | R/W | Descriptor base address |
| 0x1C | `reg_desc_tail` | R/W | Tail index |
| 0x20 | `reg_desc_head` | R | Head index (read-only) |
| 0x28 | `reg_debug` | R | See docs/AGENTS.md for bitfields |
| 0x2C | `reg_clk_cnt` | R | Free-running 100 MHz clock counter |
| 0x30 | `reg_clk_cnt_slow` | R | Clock counter ÷ 1024 |
| 0x34 | `reg_act_info` | R | act_addr from last descriptor |
| 0x38 | `reg_desc_info` | R | {8'h0, act_total_bytes[23:0]} |
| 0x3C | `reg_q8_debug` | R | Q8 core debug word |

### FSM states (see docs/AGENTS.md for full 20-state table)

0=IDLE, 1=FETCH_DESC, 3=LOAD_ACT, 5=WRITE_RES, 7=DONE,
8=LOAD_WEIGHT, 10=LOAD_SCALES, 12=COPY_ACT_TO_CORE, 13=COMPUTE,
15=READ_RES, 16=READ_RES_ACC, 17=COPY_ACC_TO_BUF, 18=TIMEOUT_ERROR

---

## Descriptor Format (32 bytes, DDR)

| Offset | Field | Size | Description |
|--------|-------|------|-------------|
| 0x00 | next_addr | 4 | Next descriptor address (0 = end) |
| 0x04 | weight_addr | 4 | Weight data address |
| 0x08 | act_addr | 4 | Activation data address |
| 0x0C | result_addr | 4 | Result write address |
| 0x10 | tensor_type | 2 | 15=CPU_OP, 0=Q8 compute |
| 0x12 | (reserved) | 2 | |
| 0x14 | num_groups | 1 | Q8 column groups (0=use GP0 reg) |
| 0x15 | (reserved) | 3 | |
| 0x18 | act_total_bytes | 4 | Total bytes to read |
| 0x1C | (reserved) | 4 | Must be 0 |

---

## Standardized Test Flow (hp_fsm_top)

This is the proven flow from `run_hp_fsm_comprehensive.tcl`:

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
mwr -force 0xF8008004 0x00000044; after 20
mwr -force 0xF8008000 0x00000005; after 20
mwr -force 0xF8008008 0x00000001; after 20
mwr -force 0xF800800C 0x00000001; after 20

# === 4. Verify GP0 access ===
set clk_cnt [read32 0x43C0002C]
if {$clk_cnt == 0} { puts "WARN: PL not clocked!" }

# === 5. Write descriptor to DDR ===
# See format table above — uses write_desc proc from run_hp_fsm_comprehensive.tcl

# === 6. Configure and start FSM ===
mwr -force 0x43C00018 <DESC_ADDR>
mwr -force 0x43C0001C 1
mwr -force 0x43C00000 1

# === 7. Poll for done (HEAD registers advances per descriptor) ===
set done 0
for {set i 0} {$i < 200} {incr i} {
    set head [read32 0x43C00020]
    if {$head >= $expected_head} { set done 1; break }
    after 50
}

# === 8. Verify results ===
set debug [read32 0x43C00028]
set status [read32 0x43C00014]

# === 9. Cleanup ===
catch {stop}; after 200
```

---

## Known Issues

### HP0 Write FIFO = 0 Blocks
Boot ROM sets `AFI0_FIFO_PARTITION=0x00000007` → write channel = 0 FIFO entries (write-once). Multi-beat write bursts silently drop data. **Workaround**: `axihp_write_master.v` uses single-beat AXI writes (AWLEN=0, AWSIZE=2 → 4 bytes). Each 64-bit word splits into two 32-bit single-beat transactions.

### AXI3 Burst Limit
PS7 HP ports are AXI3. Max burst length = 16 beats. With ARSIZE=2 (32-bit): max 64 bytes/burst. The FSM splits larger transfers across multiple bursts.

### Power-Cycle Protocol
Required before every test session. PS7 PLLs cannot be re-initialized if already locked.
1. Disconnect ALL cables, wait 60+ seconds
2. Reconnect JTAG USB only
3. **Do not** use `rst -processor` between runs — corrupts DAP irrecoverably

### XSDB Target Visibility
Digilent HS-2 detection is intermittent. If targets show 0 devices:
- Kill stale `hw_server.exe` / `cs_server.exe` processes
- Use `fpga -file` to make A9 targets visible
- Try longer `after` delays (10-15s)

### REG_DEBUG bitfield vs regs.h
The C header `vivado_integration/sw/regs.h` defines DBG_* masks matching the RTL at hp_fsm_top.v:793-804. If reading reg_debug in Tcl, decode manually:
```tcl
set dbg [read32 0x43C00028]
set state   [expr ($dbg >> 27) & 0x1F]
set rd_done [expr ($dbg >> 26) & 1]
```
