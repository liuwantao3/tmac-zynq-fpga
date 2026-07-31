# Vitis Linux Project for MicroPhase Z7-Lite (Zynq 7010)

A Vitis 2023.1 workspace for building Linux images and applications for the
tmac FPGA design. This mirrors the Xilinx Embedded Design Tutorial (2023.1)
"Building and Debugging Linux Applications for Zynq-7000 SoCs":
https://xilinx.github.io/Embedded-Design-Tutorials/docs/2023.1/build/html/docs/Introduction/Zynq7000-EDT/4-linux-for-zynq.html

## What was created

| Item | Location | Notes |
|------|----------|-------|
| Hardware platform | `matmul_bd.xsa` | Same design as `vivado_integration` (hp_fsm_top + Q8/Q5_0 cores) |
| Platform `z7_linux` | `workspace/z7_linux/` | OS=Linux, proc=`ps7_cortexa9` (cluster, required for Linux), FSBL auto-built (`fsbl.elf`), `z7_linux.xpfm` exported |
| App `hello_linux` | `workspace/hello_linux/` | "Linux Hello World" template, cross-compiled with Vitis aarch32 sysroot (no PetaLinux SDK needed). Enhanced to write a DDR verification marker + read FPGA registers via /dev/mem |
| Prebuilt Linux | `prebuilt/` | zImage, devicetree.dtb, uramdisk.image.gz, u-boot.elf, system_wrapper.bit (from the Linux-on-SD build, 2026-07-18) |
| Boot script | `scripts/boot_linux_jtag.tcl` | JTAG boot for the Vitis XSCT console (or standalone xsdb) |

## How it aligns with the Xilinx EDT 2023.1 standard

| EDT step | Standard | Here |
|----------|----------|------|
| Linux images | PetaLinux → `BOOT.BIN` + `image.ub` + `boot.scr` | Prebuilt uImage/uramdisk/dtb from the Lima VM U-Boot/kernel build (functionally equivalent; JTAG boot instead of SD) |
| Linux domain | OS=Linux, Processor=`ps7_cortexa9` | Same (this processor name is mandatory for Linux domains) |
| App | "Linux Hello World" template, SYSROOT optional | Same; built without external SYSROOT (Vitis ships the aarch32 Linux sysroot) |
| Run app from GUI | Requires **TCF agent over Ethernet** + UART login | **Not possible on this board** — no Ethernet. Verified via JTAG + DDR markers instead |

## Board constraints that change the run flow

1. **No Ethernet** (MicroPhase Z7-Lite) → Vitis "Run As → Linux Application Debug"
   cannot deploy via TCF agent. The Linux domain + app are used for
   cross-compilation; execution is done via JTAG.
2. **USB-UART on UART0 works** (CH340, MIO 14/15, 115200 8N1) but is not used for
   the run flow here: kernel console is on JTAG DCC (`hvc0`). DCC via
   `readjtaguart` is capped at ~544 B/session and
   the drain dies after ~4 sessions, so a full kernel log cannot be captured.
   For a GUI-verifiable serial flow on this same hardware, use the bare-metal
   workspace `../vitis_bm/` (Vitis GUI → Run As → Launch Hardware, console on UART0).
3. **No SD reader on Windows** → SD-card boot images are built on the Lima VM.

## Open the workspace in the Vitis GUI

1. Start Vitis 2023.1:
   `C:\Xilinx\Vitis\2023.1\bin\vitis.bat`
2. Select workspace: `D:\Users\u\tmac-zynq-fpga\vitis_linux\workspace`
3. In the Explorer view you will see `z7_linux` (platform) and `hello_linux` (app).
4. Double-click `z7_linux/platform.spr` to inspect: `linux_domain` (Linux on
   ps7_cortexa9, boot components → `vitis_linux/prebuilt`) + `zynq_fsbl`.
5. Build: right-click the platform / app → **Build Project** (hammer).
6. Program the FPGA: **Xilinx → Program FPGA** → select device `xc7z010`,
   bitstream `workspace/z7_linux/hw/matmul_bd.bit`.

## Boot Linux from the GUI

The board must be **power-cycled** first (PLL re-lock hang in ps7_init).

1. **Xilinx → XSCT Console** (integrated TCL console).
2. `source {D:/Users/u/tmac-zynq-fpga/vitis_linux/scripts/boot_linux_jtag.tcl}`
3. The script programs the bitstream, runs ps7_init + AFI, loads zImage/dtb/
   initramfs, and boots the kernel. Output shows in `dcc_boot_output.txt`.
4. Verify the kernel is alive: `pc` will have advanced, CLK_CNT keeps counting.

## Running hello_linux on the target

The compiled ELF is at `workspace/hello_linux/Debug/hello_linux.elf` (dynamic,
needs the rootfs libc). To actually execute it on Linux:

- **On the Lima VM**: add `hello_linux.elf` to the BusyBox initramfs
  (`uramdisk.image.gz`) or the ext4 partition, then reboot. When it runs it
  writes:
  - `0x1F000000 = 0x4F4C4848` ("HLLO")
  - `0x1F000004 = CLK_CNT` (FPGA free-running counter)
  - `0x1F000008 = STATUS` (HP FSM status)
  - `0x1F00000C = 0x0A424E45` ("ENB\n")
  Read these back with XSDB `mrd 0x1F000000 4` after boot. Requires
  `iomem=relaxed` in the kernel bootargs (already set) for /dev/mem access to
  0x1F000000 (kernel RAM).
- **Console path**: the USB-UART (UART0) works — attach a 115200 8N1 terminal
  and run `./hello_linux` at the login shell (kernel console currently boots to
  JTAG DCC `hvc0`).

## Rebuild from scratch (headless)

```tcl
# C:\Xilinx\Vitis\2023.1\bin\xsct.bat
setws {D:/Users/u/tmac-zynq-fpga/vitis_linux/workspace}
platform create -name z7_linux -hw {D:/Users/u/tmac-zynq-fpga/vitis_linux/matmul_bd.xsa} -proc ps7_cortexa9 -os linux -out {D:/Users/u/tmac-zynq-fpga/vitis_linux/workspace}
platform active z7_linux
domain active linux_domain
domain config -boot {D:/Users/u/tmac-zynq-fpga/vitis_linux/prebuilt}
platform generate
app create -name hello_linux -platform z7_linux -domain linux_domain -template "Linux Hello World"
app build -name hello_linux
```

Notes:
- `-proc ps7_cortexa9` (NOT `ps7_cortexa9_0`) is mandatory for a Linux domain
  on Zynq-7000.
- The "Linux Hello World" template name differs from the bare-metal one.
- device-tree-xlnx was NOT required: with a prebuilt boot image the platform
  skips Petalinux/DTS generation ("Skipping the Petalinux build as image is
  already given").
