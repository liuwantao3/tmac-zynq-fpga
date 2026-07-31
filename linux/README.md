# FPGA Accelerator Linux-on-SD Boot

**Single-machine flow (Mac):** the Lima VM builds U-Boot + kernel + initramfs
**and** `BOOT.BIN` (via a buildroot-built Xilinx `bootgen`), so the whole
SD-boot image is produced on the Mac. Windows (Vivado) is only needed to
rebuild the bitstream/FSBL if the hardware design changes.

Hardware: MicroPhase Z7-Lite (xc7z010clg400-1), UART0 MIO14/15 (CH340 USB-UART, 115200 8N1). All console output (U-Boot + kernel) is on UART0.

## Quickstart: automated build (recommended)

Single-command flow that an automated agent (or the user) can run end-to-end.

### One-time: Lima VM setup

```bash
brew install lima
limactl start --name=linux-build --cpus=8 --memory=16 template://ubuntu
limactl shell linux-build
sudo apt update && sudo apt install -y \
    gcc-arm-linux-gnueabihf build-essential flex bison bc libelf-dev libssl-dev \
    busybox-static cpio git pkg-config
exit   # back to the macOS host
```

`build_all.sh` auto-detects the host: inside the Lima VM it uses the apt
cross-compiler + libssl-dev; on the macOS host it uses the clang wrapper
(`setup_toolchain.sh`) + Homebrew binutils/openssl. **The Lima VM is the
recommended environment.**

### 1. Clone the repo (macOS host — the VM shares the host home dir)

```bash
cd ~
git clone https://github.com/liuwantao3/tmac-zynq-fpga.git
cd ~/tmac-zynq-fpga
```

The committed `linux/boot/` already contains the Windows-built
`system_wrapper.bit`, `matmul_bd.xsa`, `boot.bif`, `boot.cmd`, and the `tmac`
ARM binary. Everything else is built below.

### 2. Clone U-Boot / kernel / buildroot (inside the VM: `limactl shell linux-build`)

```bash
cd ~/tmac-zynq-fpga
bash linux/clone_repos.sh
# clones into /tmp/arm-build/{u-boot-xlnx,linux-xlnx,buildroot} (VM-local tmp)
```

### 3. Build everything (inside the VM)

```bash
cd ~/tmac-zynq-fpga
bash linux/build_all.sh          # from the repo root; artifacts land in linux/boot/
# or with explicit paths:
# bash linux/build_all.sh /tmp/arm-build ~/tmac-zynq-fpga
```

Build time ~5-15 min. Outputs in `linux/boot/` (shared with the host):
`u-boot.elf`, `boot.scr`, `uImage`, `devicetree.dtb`, `uramdisk.image.gz`,
`devicetree-jtag.dtb`, `initramfs.cpio.gz`, and — via `build_bootbin.sh`
(buildroot `host-bootgen`) — `BOOT.BIN`. `build_all.sh` does all of it; no
Windows step is needed for the SD flow.

### 4. Verify the artifacts

```bash
cd ~/tmac-zynq-fpga/linux/boot
ls -la
file uImage boot.scr uramdisk.image.gz devicetree.dtb u-boot.elf
```

Expected (approximate, from the last build):

| file | size | check |
|------|------|-------|
| `u-boot.elf` | ~1.1 MB | ARM ELF |
| `boot.scr` | ~500 B | mkimage legacy script (magic `0x27051956`) |
| `uImage` | ~4.9 MB | mkimage legacy kernel (magic `0x27051956`) |
| `devicetree.dtb` | ~17 KB | DTB |
| `uramdisk.image.gz` | ~1.3 MB | gzip (`1F 8B`) — raw gzipped cpio, no mkimage header |
| `devicetree-jtag.dtb` | ~17 KB | patched DTB (JTAG fallback only) |
| `initramfs.cpio.gz` | ~1.3 MB | gzip — same content as `uramdisk.image.gz` |
| `BOOT.BIN` | ~3.2 MB | FSBL + bitstream + U-Boot fused by buildroot `bootgen` |
| `fsbl.elf` | ~228 KB | committed (see below) — only regenerated if HW changes |

### 5. Prepare the SD card (macOS host)

```bash
diskutil list                     # find the SD disk, e.g. /dev/disk4
diskutil partitionDisk /dev/disk4 MBR FAT32 SD_BOOT 128M FAT32 SD_DATA R
```

| Partition | Type | Size | Contents |
|-----------|------|------|----------|
| 1 (`SD_BOOT`) | FAT32 | ~128 MB | `BOOT.BIN`, `uImage`, `devicetree.dtb`, `uramdisk.image.gz`, `boot.scr` |
| 2 (`SD_DATA`) | FAT32 (vfat) | Rest | `model.tmac` (~374 MB), `tmac` |

**`model.tmac` is NOT in the repo** (gitignored, ~374 MB). Ask the user for it
(it lives on the Windows machine at `models/model.tmac`) and copy it onto the
SD p2 — everything else can be built and verified without it.

### 6. Copy the SD card contents

`BOOT.BIN` was already created in step 3 by `build_bootbin.sh` (buildroot
`host-bootgen` inside the Lima VM — no Windows/Vivado needed). Copy:

| → SD p1 (`SD_BOOT`) | → SD p2 (`SD_DATA`) |
|---------------------|---------------------|
| `BOOT.BIN`, `uImage`, `devicetree.dtb`, `uramdisk.image.gz`, `boot.scr` | `model.tmac`, `tmac` |

The only reason to touch Windows is a hardware-design change: rebuild the
bitstream/FSBL there and update `linux/boot/` (see "Regenerating BOOT.BIN").

---

## Manual Build (reference — what `build_all.sh` automates)

### 1. Build U-Boot (console on UART0)

```bash
cd ~
git clone --depth=1 --branch xilinx-v2022.1 \
    https://github.com/Xilinx/u-boot-xlnx.git
cd ~/u-boot-xlnx
export CROSS_COMPILE=arm-linux-gnueabihf-

# No DCC config needed: the stock defconfig has CONFIG_ZYNQ_SERIAL=y (Cadence
# uart) and its default device tree (zynq-zc706) has stdout-path="serial0:115200n8",
# so the serial console is UART0 (CH340 USB-UART) at 115200 8N1. The stock
# CONFIG_ARM_DCC=y only adds the optional ARM DCC driver — it is not selected as
# the console because there is no "arm,dcc" DT node.
make xilinx_zynq_virt_defconfig
# Fix SPL build for Zynq 7010
sed -i 's|@dd if=$$< of=$$@ conv=block,sync bs=4 2>/dev/null;|@cp $$< $$@|' scripts/Makefile.spl
make -j$(nproc)
cp u-boot u-boot.elf
cp u-boot-spl.bin ~/tmac-zynq-fpga/linux/boot/
```

### 2. Build Linux kernel (console on UART0)

```bash
cd ~
git clone --depth=1 --branch xilinx-v2024.1 \
    https://github.com/Xilinx/linux-xlnx.git
cd ~/linux-xlnx
export CROSS_COMPILE=arm-linux-gnueabihf-

make ARCH=arm xilinx_zynq_defconfig
# UART0 console: the defconfig already enables SERIAL_XILINX_PS_UART(_CONSOLE)=y.
# Bake the command line so the console lands on ttyPS0 (used when the DT
# /chosen/bootargs is empty, e.g. the JTAG boot flow). `earlycon` (bare, no
# "=dcc") enables early boot messages on UART0 via the DT stdout-path — same
# console setup as the reference MicroPhase project (03_dma/linux/smir-top.dts:
# bootargs "earlycon, ...", stdout-path "serial0:115200n8").
./scripts/config --enable SERIAL_XILINX_PS_UART_CONSOLE
./scripts/config --set-str CMDLINE "earlycon console=ttyPS0,115200 root=/dev/ram0 rw iomem=relaxed"
make -j$(nproc) ARCH=arm olddefconfig
make -j$(nproc) ARCH=arm UIMAGE_LOADADDR=0x8000 uImage dtbs
cp arch/arm/boot/uImage ~/tmac-zynq-fpga/linux/boot/
cp arch/arm/boot/dts/zynq-zc702.dtb ~/tmac-zynq-fpga/linux/boot/devicetree.dtb
```

### 3. Build initramfs with tmac

```bash
cd ~/tmac-zynq-fpga/linux
# Build tmac-static ARM binary
arm-linux-gnueabihf-gcc -static -O2 -o tmac tmac_linux.c -lm
cp tmac boot/

# Create initramfs
mkdir -p /tmp/initramfs/{bin,dev,proc,sys,root,tmp,etc}
cp /bin/busybox /tmp/initramfs/bin/
cd /tmp/initramfs
for cmd in sh mount umount ls cat echo mknod sleep dmesg cp mv rm \
    grep sed awk hexdump md5sum devmem ps kill top free vi \
    fdisk mkfs.ext2 blkid ifconfig ping wget modprobe sync \
    reboot poweroff halt; do
    ln -sf /bin/busybox bin/$cmd
done
cp ~/tmac-zynq-fpga/linux/boot/tmac bin/

cat > init << 'INIT'
#!/bin/sh
mount -t proc none /proc
mount -t sysfs none /sys
mount -t devtmpfs none /dev
echo "=== FPGA Linux Boot — Zynq 7010 (UART0 console) ==="
echo "Console: USB-UART0 (CH340), 115200 8N1 — see docs/infrastructure.md"
echo ""
# Mount SD data partition if present (vfat or ext4, auto-detected)
mount /dev/mmcblk0p2 /root 2>/dev/null && echo "SD data mounted at /root"
echo "Ready. Run /root/tmac for FPGA test."
exec /bin/sh
INIT
chmod +x init
find . | cpio -o -H newc | gzip > ~/tmac-zynq-fpga/linux/boot/uramdisk.image.gz
```

### 4. Deploy U-Boot files

```bash
cp ~/u-boot-xlnx/u-boot.elf ~/tmac-zynq-fpga/linux/boot/
cp ~/u-boot-xlnx/u-boot-spl.bin ~/tmac-zynq-fpga/linux/boot/
```

### 5. Build boot.scr (auto-boot script)

`xilinx_zynq_virt_defconfig` sets `CONFIG_DISTRO_DEFAULTS=y`, so U-Boot
automatically runs `boot.scr` from the FAT32 partition — no interactive prompt
needed. Generate it from the committed `linux/boot/boot.cmd` with U-Boot's own
`mkimage` (do **not** use kernel u-boot-tools):

```bash
cd ~/u-boot-xlnx
./tools/mkimage -A arm -T script -C none -n "Boot" \
    -d ~/tmac-zynq-fpga/linux/boot/boot.cmd \
    ~/tmac-zynq-fpga/linux/boot/boot.scr
```

(`linux/build_all.sh` does this automatically.)

### 6. Build BOOT.BIN (Mac, buildroot bootgen — no Windows)

`build_all.sh` runs `linux/build_bootbin.sh` automatically. It builds Xilinx
`bootgen` with buildroot's `host-bootgen` package (into
`/tmp/arm-build/buildroot/output/host/bin/bootgen`) and fuses:

```
the_ROM_image:
{
    [bootloader] fsbl.elf
    system_wrapper.bit
    u-boot.elf
}
```

To run it standalone:

```bash
bash linux/build_bootbin.sh /tmp/arm-build ~/tmac-zynq-fpga
```

`fsbl.elf` is committed in `linux/boot/` (built from `matmul_bd.xsa`); it only
needs regenerating if the hardware design changes (see the next section).

---

## Windows (only if the hardware design changes)

The Mac/Lima-VM flow above produces `BOOT.BIN` itself, so Windows/Vivado is only
needed to rebuild `system_wrapper.bit` + `fsbl.elf` when the PL design changes
(and then re-run `build_bootbin.sh` on the Mac).

**Prerequisites:** Vivado 2023.1.

### Step 1: Ensure boot files

```
linux/boot/
├── system_wrapper.bit   ← committed (built in Vivado on Windows)
├── matmul_bd.xsa        ← committed (hardware handoff)
├── boot.bif             ← committed (bootgen config)
├── boot.cmd             ← committed (U-Boot auto-boot script source)
├── fsbl.elf             ← committed (built from matmul_bd.xsa)
├── u-boot.elf           ← Mac build (from build_all.sh)
├── boot.scr             ← generated by mkimage from boot.cmd (Mac build)
├── uImage               ← Mac build
├── devicetree.dtb        ← Mac build
├── uramdisk.image.gz     ← Mac build
└── tmac                 ← Mac build
```

### Step 2: Rebuild FSBL (only if matmul_bd.xsa changes)

The committed `fsbl.elf` is built from `matmul_bd.xsa`; regenerable via XSCT:

```tcl
hsi::open_hw_design linux/boot/matmul_bd.xsa
hsi::generate_app -hw linux/boot/matmul_bd.xsa -os standalone -proc ps7_cortexa9_0 -app zynq_fsbl
# Copy fsbl.elf from the generated SDK project to linux/boot/
```

### Step 3: Windows bootgen fallback (optional)

`build_bootbin.sh` on the Mac is the primary path. If you prefer, the same
command works on Windows:

```cmd
cd D:\Users\u\tmac-zynq-fpga\linux\boot
C:\Xilinx\Vivado\2023.1\bin\bootgen.bat -image boot.bif -o BOOT.BIN -w
```

`boot.bif` contents (FSBL path — SPL is **not** used; U-Boot is loaded from
`u-boot.elf` by the FSBL):
```
the_ROM_image:
{
    [bootloader] fsbl.elf
    system_wrapper.bit
    u-boot.elf
}
```

### Step 4: Boot the SD

1. Power-cycle the board (required — PLL re-init hangs on warm reset)
2. Insert SD, set boot mode jumper **J1** to SD
3. Connect the USB-UART cable and open a 115200 8N1 serial terminal (PuTTY,
   COM port of the CH340). All U-Boot + kernel console output appears here.
4. Power on — U-Boot (distro boot) auto-runs `boot.scr` from the FAT32
   partition, loads Linux, runs initramfs
5. Login shell at the initramfs prompt (see U-Boot manual boot below if auto-boot fails)

### U-Boot Manual Boot (if auto-boot fails)

These are exactly the commands embedded in `linux/boot/boot.cmd`:

```
U-Boot> fatload mmc 0 0x3000000 uImage
U-Boot> fatload mmc 0 0x2A00000 devicetree.dtb
U-Boot> fatload mmc 0 0x2000000 uramdisk.image.gz
U-Boot> setenv bootargs "console=ttyPS0,115200 root=/dev/ram0 rw iomem=relaxed"
U-Boot> bootm 0x3000000 0x2000000 0x2A00000
```

---

## JTAG Boot (no SD card, fallback for bring-up)

SD boot (above) is the primary path. This JTAG hand-boot is the fallback for
FPGA/DDR bring-up without shuffling SD cards. Direct JTAG boot (bitstream →
PS7 init → AFI → kernel/DTB/initrd → Linux) is handled by the Vitis Linux
workspace:

```tcl
# From the Vitis GUI XSCT console (power-cycle the board first):
source D:/Users/u/tmac-zynq-fpga/vitis_linux/scripts/boot_linux_jtag.tcl
```

Or standalone:

```powershell
C:\Xilinx\Vivado\2023.1\bin\xsdb.bat D:\Users\u\tmac-zynq-fpga\vitis_linux\scripts\boot_linux_jtag.tcl
```

This loads the bitstream from `vitis_linux/workspace/z7_linux/hw/`, runs
ps7_init, programs AFI (HP0), loads zImage/DTB/initramfs from
`vitis_linux/prebuilt/` to DDR, and boots the kernel (r0=0 r1=~0 r2=dtb pc=zImage).

Because this is a U-Boot-less hand boot, the kernel finds the initramfs through
the DTB: `devicetree-jtag.dtb` (built by `patch_dtb_initrd.py`) carries
`/chosen/linux,initrd-start/end` pointing at the raw gzipped cpio
(`initramfs.cpio.gz`, no U-Boot legacy header) loaded at 0x03000000.
Kernel console is on the USB-UART0 (the DTB `/chosen/bootargs` is empty, so the
kernel uses the baked-in `CONFIG_CMDLINE=console=ttyPS0,115200 ...`). Open a
115200 8N1 terminal (PuTTY) on the CH340 COM port to see it.

---

## Console: UART0 (default) vs JTAG DCC (removed)

| Feature | UART0 (used) | DCC (removed) |
|---------|--------------|---------------|
| Hardware | CH340 USB-UART (works) | JTAG (Digilent HS-2) |
| Speed | 115200 baud (~11 KB/s) | ~200-500 KB/s |
| Console | `ttyPS0` (kernel) / U-Boot serial | `hvc0` |
| Config | kernel `CONFIG_CMDLINE=console=ttyPS0,115200`; U-Boot stock defconfig (`stdout-path=serial0`) | kernel `earlycon=dcc console=hvc0`; U-Boot `CONFIG_ARM_DCC=y` |
| Capture | Serial terminal (PuTTY) | `readjtaguart -start` in XSDB |

The CH340 USB-UART works (UART0, MIO 14/15, 115200 8N1) — see AGENTS.md Key
Decision #16 and `docs/infrastructure.md`. The DCC console was adopted on
2026-07-30 while the UART was wrongly believed broken; it is capped (~544 B/session)
and the `readjtaguart` drain dies after ~4 sessions, so it has been removed from
the build. All console output is now on the USB-UART.
