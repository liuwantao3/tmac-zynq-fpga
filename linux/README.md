# FPGA Accelerator Linux-on-SD Boot

**Single-machine flow (Windows WSL):** the Linux kernel + device tree +
initramfs for the MicroPhase Z7-Lite SD boot are built inside WSL (Ubuntu
24.04). `BOOT.BIN` is fused from the committed `fsbl.elf` + `system_wrapper.bit`
+ `u-boot.elf` with Vivado's `bootgen.bat` (all three inputs are committed, so
`BOOT.BIN` only needs regenerating if the hardware changes).

**The verified-working boot artifacts are committed** in `linux/boot/` (and
`linux/scripts/` holds the JTAG bring-up helpers). Rebuilding is only needed
when the kernel/DTB/initramfs sources change.

Hardware: MicroPhase Z7-Lite (xc7z010clg400-1), UART0 MIO14/15 (CH340 USB-UART,
115200 8N1). All console output (U-Boot + kernel) is on UART0.

## Quickstart: build (WSL, Ubuntu 24.04)

### 1. One-time toolchain setup (inside WSL)

```bash
sudo apt update && sudo apt install -y \
    gcc-arm-linux-gnueabihf build-essential flex bison bc libelf-dev libssl-dev \
    u-boot-tools device-tree-compiler cpio wget
```

### 2. Clone the sources (kernel for DTB + uImage; U-Boot only for mkimage/dts)

```bash
cd ~
git clone https://github.com/liuwantao3/tmac-zynq-fpga.git
cd ~/tmac-zynq-fpga
bash linux/clone_repos.sh        # clones into ~/linux-xlnx + ~/u-boot-xlnx
```

### 3. Build the boot artifacts

```bash
cd ~/tmac-zynq-fpga
bash linux/build_wsl.sh          # from the repo root; artifacts land in linux/boot/
```

`build_wsl.sh` does, in order:
1. Configures the kernel (`xilinx_zynq_defconfig` + UART0 console + `DEBUG_LL`
   for early boot serial — see below).
2. Patches the zc702 DTS for Z7-Lite hardware (SDHCI CD/WP, GPIO pinctrl,
   gpio-keys/leds — see `docs/z7lite-vs-zc702.md`).
3. Builds `uImage` + `devicetree.dtb`.
4. Builds a 32-bit ARM static busybox from source.
5. Assembles the initramfs (with `/dev/console` nodes + `setsid`/`cttyhack`).
6. Wraps it as `uramdisk.image.gz`; emits raw `initramfs.cpio.gz`.
7. Builds `devicetree-jtag.dtb` (JTAG hand-boot variant).
8. Copies the raw `zImage` to `linux/boot/` for the JTAG hand-boot script.

### 4. Verify the artifacts

```bash
cd ~/tmac-zynq-fpga/linux/boot
ls -la
file uImage boot.scr uramdisk.image.gz devicetree.dtb
```

Expected (verified 2026-08-11):

| file | size | check |
|------|------|-------|
| `u-boot.elf` | 1,063,480 B | ARM ELF, UART0 console (committed) |
| `boot.scr` | 543 B | mkimage legacy script, loads uImage @ 0x03000000 |
| `uImage` | 4,892,736 B | mkimage legacy kernel (debug build, `DEBUG_LL`) |
| `devicetree.dtb` | 16,619 B | Z7-Lite patched DTB (see `docs/z7lite-vs-zc702.md`) |
| `devicetree-jtag.dtb` | 16,688 B | Z7-Lite DTB + `/chosen/linux,initrd-*` |
| `uramdisk.image.gz` | 1,327,882 B | U-Boot ramdisk (32-bit busybox initramfs) |
| `initramfs.cpio.gz` | 1,327,818 B | raw gzipped cpio (JTAG hand-boot) |
| `BOOT.BIN` | 3,168,704 B | FSBL + bitstream + U-Boot (from `bootgen.bat`) |
| `fsbl.elf` | 437,552 B | committed (SD-capable, `-DFSBL_DEBUG_INFO`) |

### 5. Prepare the SD card (Windows)

The SD card is a **single FAT32 partition** (label `SD_BOOT`):

| File | Purpose |
|------|---------|
| `BOOT.BIN` | FSBL + bitstream + U-Boot |
| `uImage` | Linux kernel |
| `devicetree.dtb` | Z7-Lite device tree |
| `uramdisk.image.gz` | initramfs (U-Boot ramdisk) |
| `boot.scr` | auto-boot script |
| `uEnv.txt` | `boot_targets=mmc0` (skip Ethernet polling) |
| `tmac` | FPGA test binary |
| `model.tmac` | model weights (NOT in repo; on the Windows machine) |

Copy these to the SD root, insert, set boot jumper **J1** to SD, power on.
U-Boot auto-runs `boot.scr`; no typing needed.

---

## Boot flow (verified 2026-08-11)

```
BootROM (MIO[8:6]=110=SD) → FSBL InitSD("BOOT.BIN") → PCAP loads bitstream
→ U-Boot (UART0 console) → distro boot → boot.scr → fatload uImage/dtb/ramdisk
→ bootm → Linux 6.6.0 → initramfs → /bin/sh shell
```

The kernel boots to an interactive shell (`~ #`) on the CH340 USB-UART.
FPGA registers are at `0x43C00000` (test with `devmem 0x43C0002C` for the
free-running clock counter).

### U-Boot Manual Boot (if auto-boot fails)

Exactly what `linux/boot/boot.cmd` runs:

```
U-Boot> fatload mmc 0 0x3000000 uImage
U-Boot> fatload mmc 0 0x2A00000 devicetree.dtb
U-Boot> fatload mmc 0 0x2000000 uramdisk.image.gz
U-Boot> setenv bootargs "console=ttyPS0,115200 root=/dev/ram0 rw iomem=relaxed"
U-Boot> bootm 0x3000000 0x2000000 0x2A00000
```

---

## Regenerating BOOT.BIN (Windows, only if HW changes)

`BOOT.BIN` is **not committed** (it is fused from the three committed inputs).

```cmd
cd D:\Users\u\tmac-zynq-fpga\linux\boot
C:\Xilinx\Vivado\2023.1\bin\bootgen.bat -image boot.bif -o BOOT.BIN -w
```

`boot.bif` contents (FSBL path — SPL is **not** used):

```
the_ROM_image:
{
    [bootloader] fsbl.elf
    system_wrapper.bit
    u-boot.elf
}
```

The committed `fsbl.elf` is built from `matmul_bd.xsa` via the standard Vitis
`platform create` + `platform generate` flow (see AGENTS.md Key Decision #23;
two gating pitfalls documented there and in `docs/`).

---

## U-Boot rebuild (only if the console/boot behavior changes)

The committed `u-boot.elf` already has the UART0 console, `CONFIG_OF_EMBED=y`,
`DEBUG_UART_ZYNQ` early output, and `CONFIG_SYS_L2CACHE_OFF=y` (the zc702 DTB
routes `serial0` to UART0). To rebuild:

```bash
cd ~/u-boot-xlnx
export CROSS_COMPILE=arm-linux-gnueabihf-
make xilinx_zynq_virt_defconfig
echo 'CONFIG_DEFAULT_DEVICE_TREE="zynq-zc702"' >> .config
echo 'CONFIG_OF_EMBED=y' >> .config
echo 'CONFIG_DEBUG_UART=y' >> .config
echo 'CONFIG_DEBUG_UART_ZYNQ=y' >> .config
echo 'CONFIG_DEBUG_UART_BASE=0xE0000000' >> .config
echo 'CONFIG_DEBUG_UART_CLOCK=100000000' >> .config
echo 'CONFIG_SYS_L2CACHE_OFF=y' >> .config
make olddefconfig
make -j$(nproc) u-boot
cp u-boot ~/tmac-zynq-fpga/linux/boot/u-boot.elf
```

## About the DEBUG_LL kernel

The committed `uImage` is built with `CONFIG_DEBUG_LL=y` + `EARLY_PRINTK=y` +
`DEBUG_ZYNQ_UART0=y`. These make the kernel print to UART0 *before* the console
driver exists — essential for diagnosing the 2026-08-11 "Starting kernel..."
hang (which turned out to be a DTB problem, see `docs/z7lite-vs-zc702.md`).
They are harmless at runtime; keep them on for future bring-up.

## JTAG boot note

`linux/scripts/boot_linux_jtag.tcl` (U-Boot-less hand boot) has **never been
verified on hardware** — all bring-up was done via SD boot (either the SD
auto-boot or U-Boot loaded over JTAG via `linux/scripts/boot_kernel_via_uboot_jtag.tcl`).
The `devicetree-jtag.dtb` + `initramfs.cpio.gz` artifacts are generated for it
and kept consistent, but treat SD boot as the only proven path.

## Vitis GUI workspace (optional, for cross-compiling Linux apps)

The Vitis 2023.1 GUI can be used to cross-compile a Linux userspace app against
the aarch32 sysroot, but the Z7-Lite has **no Ethernet**, so the standard
"Run As → Linux Application Debug" (TCF agent) flow is impossible. The Linux
domain/app is used for cross-compilation only; execution is done on the SD
rootfs (see "initramfs with tmac" above) or via the bare-metal workspace
`../vitis_bm/` (GUI console on UART0).

The workspace is **regenerable** (never committed) and boots the committed
`linux/boot/` artifacts. To recreate it headlessly:

```tcl
# C:\Xilinx\Vitis\2023.1\bin\xsct.bat
setws {D:/Users/u/tmac-zynq-fpga/vitis_linux/workspace}
platform create -name z7_linux -hw {D:/Users/u/tmac-zynq-fpga/linux/boot/matmul_bd.xsa} -proc ps7_cortexa9 -os linux -out {D:/Users/u/tmac-zynq-fpga/vitis_linux/workspace}
platform active z7_linux
domain active linux_domain
domain config -boot {D:/Users/u/tmac-zynq-fpga/linux/boot}
platform generate
app create -name hello_linux -platform z7_linux -domain linux_domain -template "Linux Hello World"
app build -name hello_linux
```

Notes:
- `-proc ps7_cortexa9` (NOT `ps7_cortexa9_0`) is mandatory for a Linux domain.
- device-tree-xlnx is NOT required: with a prebuilt boot image the platform
  skips PetaLinux/DTS generation.
- The `hello_linux` app template writes DDR markers (`0x1F000000 = "HLLO"`,
  CLK_CNT at `0x1F000004`, STATUS at `0x1F000008`) readable via XSDB `mrd`.
- JTAG bring-up helpers live in `linux/scripts/` (boot/debug/verify tools).
