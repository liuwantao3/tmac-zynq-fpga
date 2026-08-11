# Z7-Lite vs ZC702 Reference Board — Hardware Differences

The Linux/U-Boot bring-up for the MicroPhase Z7-Lite (xc7z010clg400-1) uses the
Xilinx **zynq-zc702** device tree as its base. The ZC702 is a different board,
and its DTS encodes assumptions that **silently break** the Z7-Lite. Every one
of the 2026-08-11 kernel-bring-up bugs traced back to one of these mismatches.

This document is the canonical delta. When editing the DTS/DTB, re-check every
row against the Z7-Lite's actual MIO wiring (see `docs/infrastructure.md` for
the verified MIO map).

## MIO pin assignments — the single source of truth

| MIO | Z7-Lite (actual) | zc702 DTS says | Conflict |
|-----|------------------|----------------|----------|
| 0 | — (unused) | `sdio0_cd` (SD card detect) | none (unused pin), but see note |
| 7-13 | — (unused) | `gpio0` mux group | none (harmless) |
| 14 | **UART0 TX** (CH340) | `gpio0` mux group + `gpio-keys` sw13 | **UART0 TX stolen** |
| 15 | **UART0 RX** (CH340) | `sdio0_wp` (SD write-protect) | **UART0 RX stolen** |
| 40-45 | **SDIO0** (SD card, function 3) | `sdio0_2_grp` (sdio0) | matches (function 2) |
| 50-51 | — (unused) | `i2c0` scl/sda (i2c0_10_grp) | none (i2c0 not on board) |

> **MIO 14/15 are the lifeblood of bring-up** — they are the CH340 USB-UART
> console. Anything in the DTS that claims MIO 14 or 15 for another function
> disables the console.

## The three DTS fixes (each one a distinct symptom)

### 1. sdhci0 CD/WP pinctrl → SDHCI driver hang

**Symptom:** kernel hangs right after `sdhci-pltfm: SDHCI platform and OF
driver helper`; no `mmc0: SDHCI controller on...` line ever appears.

**Cause:** the zc702 `&sdhci0` node carries
`pinctrl-0 = <&pinctrl_sdhci0_default>` which remuxes **MIO 15 → sdio0_wp**
and MIO 0 → sdio0_cd. On the Z7-Lite SDIO0 has **CD/WP off** (block-design
`PCW_SD0_..._CD/WP` not wired), and MIO 15 is UART0 RX. The pinctrl driver
stalls probing the CD/WP MIO groups.

**Fix (DTS):**
```
&sdhci0 {
	bootph-all;
	status = "okay";
	xlnx,has-cd = <0x0>;
	xlnx,has-power = <0x0>;
	xlnx,has-wp = <0x0>;      /* <-- no pinctrl, no CD/WP pins */
};
```
This matches the MicroPhase reference `smir-top.dts` (`xlnx,has-cd/power/wp = <0>`).

### 2. gpio0 pinctrl → dead UART RX (console shows prompt, no input)

**Symptom:** kernel boots, initramfs shell prints `~ #`, but **keyboard input
does nothing**. `devmem 0xE000002C` (UART SR) reads `0xA` = RX empty + TX
empty even while typing. U-Boot receives input fine on the same UART.

**Cause:** the `gpio@e000a000` node has
`pinctrl-0 = <&pinctrl_gpio0_default>`, and that group remuxes
**MIO 7-14 → gpio0** (mux_val=0, clearing the UART0 function selector 0x70 on
MIO 14). The GPIO driver applies it during probe, so UART0 TX is remuxed away.
(RX on MIO 15 was already dead from the sdio0_wp remux in fix #1.)

**Fix (DTS):** remove the pinctrl from the GPIO node:
```
&gpio0 {
};
```

### 3. gpio-keys + leds nodes → more MIO 14/15 stealing

**Symptom:** (same UART death as #2; fixed together)

**Cause:** the zc702 DTS has `gpio-keys` (sw13 on **GPIO 14**) and `leds`
(led-ds23 on GPIO 10). GPIO 14 = MIO 14 = UART0 TX. The `gpio-keys` driver
requests GPIO 14, reinforcing the remux.

**Fix (DTS):** delete both nodes (they drive switch/LED hardware that is not
on the Z7-Lite).

## What was NOT the problem (checked and ruled out)

| Hypothesis | Verdict |
|------------|---------|
| Kernel load address (0x00008000 vs 0x00200000 vs 0x00108000) | All boot the kernel once it is a valid zImage; the hang was the *DTB*, not the address |
| Ramdisk / initramfs format | Was a problem (AArch64 busybox, see below) but *not* the DTB hang |
| UART0 clock / baud | OK — xuartps finds `base_baud = 6249999` from the 100 MHz UART clock |
| Missing `/dev/console` in initramfs | Separate initramfs issue (keyboard no input), distinct from the DTB mux issue |

## Related: initramfs issues (not DTB, but same bring-up session)

1. **AArch64 busybox → `Failed to execute /init (error -8)` (ENOEXEC).** The
   stock buildroot cpio shipped a 64-bit busybox; the Zynq-7010 is ARMv7
   32-bit. Rebuilt busybox from source with `arm-linux-gnueabihf-gcc`
   (`CONFIG_STATIC=y`, `CONFIG_TC=n` — new kernel headers removed `TCA_CBQ_*`).
2. **Empty `/dev` in cpio → shell prompt but no input.** The kernel opens
   `/dev/console` for the init process *before* `/init` runs; if the node is
   missing, stdin becomes `/dev/null`. Added `dev/console` (5,1), `dev/null`
   (1,3), `dev/tty` (5,0) to the cpio via `gen_init_cpio` (no root needed).
3. **`setsid: not found` → `Attempted to kill init!`.** The busybox applets
   `setsid` and `cttyhack` must have symlinks in `/bin`; then
   `setsid cttyhack sh` gives the shell its controlling terminal (the
   documented fix for "can't access tty; job control turned off").

## How to verify after a DTB edit

Boot and check:
- Console TX works: U-Boot + kernel banner on CH340.
- Console RX works: type at the `~ #` prompt; `devmem 0xE000002C` should show
  bit1 (RX empty) **cleared** while a key is held.
- SD card works: `mmc0: SDHCI controller on e0100000.mmc` then
  `mmcblk0: mmc0:...` then `mmcblk0: p1`.

If any one of these breaks, re-audit the three MIO rows in the table above.
