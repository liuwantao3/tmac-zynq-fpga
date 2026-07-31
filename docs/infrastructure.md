# Board Infrastructure & Lessons Learned (MicroPhase Z7-Lite / Zynq-7010)

This document treats the board's infrastructure subsystems — the **serial console
(UART0)**, **DDR3 / AXI HP0**, **PS7 clocks & PLL**, and **JTAG/DAP** — as
first-class components. Each section records verified facts, the failure modes
encountered, and the lessons learned. If something stops working, read this first.

> Rule of thumb from the UART episode: **verify the hardware with a known-good
> reference before blaming it or working around it in software.** A debug-channel
> workaround built on an unverified "broken hardware" assumption becomes a second
> bug on top of the first.

---

## 1. USB-UART0 Serial Console — MAJOR INFRASTRUCTURE

UART0 is the primary console for this board (bare-metal and, later, Linux
`ttyPS0`). The USB-UART (CH340 bridge, MIO 14/15) is confirmed **working** on the
physical hardware.

### Verified facts

| Item | Value |
|------|-------|
| UART0 base | `0xE0000000` |
| MIO | 14 (TX), 15 (RX) |
| Speed / format | 115200 baud, 8N1 |
| Reference clock | 100 MHz (PS7 UART ref clk) |
| Register map | see below |
| Bridge | CH340 on-board USB-UART (verified working via reference `03_dma` project) |

### Register map (xuartps)

| Offset | Word | Name | Notes |
|--------|------|------|-------|
| `0x00` | `uart[0]` | CR | Control: `RXRST=0x01`, `TXRST=0x02`, `RX_EN=0x04`, `TX_EN=0x10`, `TX_DIS=0x20` |
| `0x04` | `uart[1]` | MR | Mode: `0x20` = 8N1, normal mode |
| `0x18` | `uart[6]` | BAUDGEN | Clock divisor (see baud math) |
| `0x2C` | `uart[11]` | SR | Status: bit 4 (`0x10`) = TXFULL |
| `0x30` | `uart[12]` | FIFO | Transmit/Receive FIFO |
| `0x34` | `uart[13]` | BAUDDIV | Clock divisor (see baud math) |

Baud math: `baud = f_ref / ((BAUDGEN+1) × (BAUDDIV+1))`.
With `BAUDGEN=124`, `BAUDDIV=6`: `100e6 / (125 × 7) ≈ 114 286 ≈ 115200` (within
UART tolerance). **These exact values are also what `ps7_init` programs.**

`ps7_init` (via `ps7_peripherals_init_data_3_0`) already configures UART0 correctly
(`CR=0x17`). Bare-metal code must nevertheless program it itself to be
self-contained (SD boot, standalone runs, GUI launches without ps7_init).

### The failure (2026-07-30/31) — why the old setup was silent

Three independent bugs, any one of which alone would break output:

| # | Bug | Consequence |
|---|-----|-------------|
| 1 | `uart_put*()` bodies were **JTAG DCC wrappers** | **Root cause:** nothing was ever written to the TX FIFO — output went to DCC instead. (This was a *workaround for a wrongly-assumed dead CH340*.) |
| 2 | Final `CR` write was `0x20` = **TX_DIS** (TX_EN is bit 4 = `0x10`) | Transmitter explicitly disabled after init |
| 3 | Baud written to `uart[8]`/`uart[9]` → offsets `0x20`/`0x24` (RXWM/MODEMCR) instead of BAUDGEN `0x18` / BAUDDIV `0x34` | Baud registers never touched — masked because ps7_init had already set them correctly |

Diagnosis path that eventually proved the CH340 was fine: running the reference
MicroPhase project `D:\Users\u\microphase-z7\03_dma\arm` in the Vitis GUI printed
"Hello from MicroPhase Z7-Lite..." on the same USB-UART with an identical PS7
config. After that, re-auditing `uart_init()` against `xuartps_hw.h` found bugs
#2 and #3, and removing the DCC redirect fixed #1.

### Canonical working sequence (as fixed in `tmac_baremetal.h`)

```c
uart[0] = 0x00000000;            // CR: disable all
uart[0] = 0x00000001;            // CR: RXRST
uart[0] = 0x00000002;            // CR: TXRST
uart[0] = 0x00000000;            // CR: idle
uart[1] = 0x00000020;            // MR: 8N1, normal mode
uart[6] = 124;                   // BAUDGEN (0x18)
uart[13] = 6;                    // BAUDDIV (0x34)
uart[0] = 0x00000014;            // CR: RX_EN(0x04) | TX_EN(0x10)   <-- NOT 0x20!
```

Transmit (polling):

```c
while (uart[11] & 0x10);         // SR bit4 = TXFULL: wait for FIFO space
uart[12] = (uint32_t)(unsigned char)c;   // FIFO (0x30)
```

### Testable entry points

| Flow | How to run | What you see |
|------|-----------|--------------|
| Vitis GUI (bare-metal) | `vitis.bat` workspace `vitis_bm` → `tmac_serial` → Run As → Launch Hardware | banner + FPGA regs + `tick` on USB-UART terminal |
| Headless XSDB | `xsdb.bat vitis_bm\scripts\run_serial.tcl` (loads `sw/uart_test.elf`) | banner + tick loop |
| Reference proof | MicroPhase `03_dma/arm` in Vitis GUI | prints on the same UART |

### JTAG DCC — the fallback console (use sparingly)

DCC console capture (`readjtaguart`) works but is unreliable for long output:
capped at ~544 B/session and the drain dies after ~4 sessions. It is **not** a
substitute for the UART. Kernel console migration UART0 (`ttyPS0`) is the planned
Linux increment.

---

## 2. DDR3 SDRAM (PS7 controller, 512 MB)

| Item | Value |
|------|-------|
| Part | Micron MT41J256M16 RE-125 (x16, 4 Gbit) |
| Speed | DDR3-1066F (533 MHz core) |
| PS↔PL addressing | Shared — PS writes at address X are visible to PL at the same X (verified by `test_addr_map2.tcl`) |
| Program base | `0x00100000` (link.ld) |

Lessons learned:

- **HP0 is effectively 32-bit on this board** despite `PCW_S_AXI_HP0_DATA_WIDTH=64`.
  `ARSIZE=3` reads return `RDATA[63:32]=0` (upper half always 0 with x16 DDR3);
  `AFI0_CTRL[7:6]` 64-bit enable is read-only. Reads use `ARSIZE=2`. **Writes are
  fine with `AWSIZE=3`** — the HP port performs two internal 32-bit accesses.
- Scratch buffers must live in the low ~500 MB of DDR. The range
  `0x1F000000–0x1F004000` is verified working; the `0x17E00000` range showed
  access issues.
- **Power-cycle before ps7_init** (see §4) or DDR calibration never completes and
  every subsequent read returns 0.

---

## 3. AXI HP0 High-Performance Port

- AXI3 — **max 16 beats/burst**. `ARLEN > 15` is silently rejected (read master
  never sees `RLAST`). All bursts hard-limited to 16 beats.
- AFI0 registers at `0xF800_8000` (NOT `0xF800_9000` as initially documented).
- Verified AFI sequence:

  ```
  mwr 0xF8000008 0x0000DF0D    # unlock SLCR
  mwr 0xF8000910 0x0000000F    # LVL_SHFTR_EN
  mwr 0xF8008000 0x00000005    # AFI0_CTRL (enable + slverr)
  mwr 0xF8008004 0x00000044    # AFI0_PART (R:4, W:4 entries)
  mwr 0xF8008008 0x00000001    # AFI0_WRCHAN
  mwr 0xF8000004 0x0000767B    # lock SLCR
  ```

- Read-master handshake gotcha: `rd_ready <= rd_valid` (delayed handshake), not a
  free-running `rd_ready=1`, otherwise the 0-cycle `rvalid` pulse self-clears in
  the read master's PRESENT state.

---

## 4. PS7 Clocks & PLL (ps7_init)

- `ps7_pll_init_data_3_0` **hangs if the PLLs are already configured** (e.g. from a
  prior session). The reset/re-lock sequence cannot re-lock an already-locked PLL.
- After a hang, ALL subsequent ps7_init attempts also fail, HP0 reads return 0,
  and the PS7 AHB interconnect is in an inconsistent state.
- **Workaround / rule: always power-cycle the board before any ps7_init via XSDB.**
  `rst -processor` is NOT sufficient.

---

## 5. PL Clock (FCLK_CLK0)

- DAP writes to `FPGA_CLK_CTRL[7]` (`0xF8000170`) are ignored — the register is
  locked to secure mode. **The ARM boot code must enable FCLK_CLK0** before the
  PL clock can run. Verify via `REG_CLK_CNT` (0x43C00000 + 0x2C) advancing.

---

## 6. JTAG / DAP

| Hazard | Consequence | Recovery |
|--------|-------------|----------|
| `rst -processor` | Corrupts DAP irreversibly (reads `0xF0000021`) | Power-cycle (disconnect all cables, wait 60+ s — decoupling caps hold charge) |
| ps7_init hang | PS7 AHB broken, reads return 0 | Power-cycle |
| AP transaction error `0x30000001` | Transient (often after `fpga -file`) | Retry / DP CTRL/STAT clear via `mwr -dp 0x4 0x00000000` |

---

## 7. Cross-cutting Lessons Learned

1. **Verify the hardware before working around it.** The CH340 "broken" assumption
   cost a week and spawned the DCC redirect (which then became the actual root
   cause of the silent console).
2. **Register programming bugs come in three flavors** — wrong bit mask (TX_EN vs
   TX_DIS), wrong offset (word index ≠ byte offset), and wrong path (data never
   reaching the hardware at all). Audit all three when a peripheral is silent.
3. **`ps7_init` masks your bugs.** It configures the peripherals correctly, so a
   wrong bare-metal init may appear to work at first — and only break when the app
   runs standalone. Always verify the bare-metal init is self-sufficient.
4. **Check the disassembly.** The register constants you *intend* to write must
   appear in the compiled object (e.g. `mov r1, #124`, `mov r1, #20`, FIFO `#48`).
5. **Document infrastructure facts with their proof** (see §1 testable entry
   points) so a future "dead port" report can be closed quickly.
