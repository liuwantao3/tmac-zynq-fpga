# FPGA Accelerator Linux-on-SD Boot

**Built and tested in Lima ARM VM (Ubuntu 24.04 aarch64) on macOS.**

## Build Results (2026-07-18)

| File | Size | Source |
|------|------|--------|
| `u-boot.img` | 1.2 MB | U-Boot image (loaded by SPL from SD) |
| `u-boot-spl.bin` | 121 KB | SPL (loaded by FSBL, runs from OCM) |
| `uImage` | 4.6 MB | Linux 6.6.0-xilinx (xilinx_zynq_defconfig) |
| `devicetree.dtb` | 17 KB | zynq-zc702.dts (built from kernel tree) |
| `uramdisk.image.gz` | 1.3 MB | BusyBox initramfs (79 applets + tmac) |
| `tmac` | 483 KB | Static ARM32, compiled from tmac_linux.c |

All 5 files at `~/arm-build/` on macOS. Copy to Windows for SD card creation.

## Hardware

- **Board**: MicroPhase Z7-Lite (Zynq 7010, xc7z010clg400-1)
- **DDR3**: 512 MB at 0x00100000–0x20000000
- **PL clock**: 100 MHz (FCLK_CLK0)
- **FPGA IP**: hp_fsm_top at AXI4-Lite 0x43C00000 (GP0)
- **UART**: UART0 at 0xE0000000, MIO 14/15, 115200 baud
- **SD**: SD0 at 0xE0100000, MIO 40-45
- **/dev/mem**: CONFIG_DEVMEM=y enabled in kernel

## Build Environment (Reproducible)

Built in Lima ARM Ubuntu VM on Apple Silicon Mac:

```bash
# 1. Start VM (one-time setup)
brew install lima
limactl start --name=linux-build --cpus=8 --memory=16 template://ubuntu
limactl shell linux-build

# 2. Inside VM — install tools
sudo apt update && sudo apt install -y gcc-arm-linux-gnueabihf \
    build-essential flex bison bc libelf-dev libssl-dev git

# 3. Clone repos
cd ~ && git clone --depth=1 --branch xilinx-v2022.1 \
    https://github.com/Xilinx/u-boot-xlnx.git
git clone --depth=1 --branch xilinx-v2024.1 \
    https://github.com/Xilinx/linux-xlnx.git

# 4. Build U-Boot
cd ~/u-boot-xlnx
export CROSS_COMPILE=arm-linux-gnueabihf-
sed -i 's|@dd if=$$< of=$$@ conv=block,sync bs=4 2>/dev/null;|@cp $$< $$@|' scripts/Makefile.spl
make xilinx_zynq_virt_defconfig && make -j8

# 5. Build Linux kernel
cd ~/linux-xlnx
make ARCH=arm xilinx_zynq_defconfig
make -j8 ARCH=arm UIMAGE_LOADADDR=0x8000 uImage dtbs

# 6. Build initramfs (BusyBox + tmac)
sudo apt install -y busybox-static cpio
mkdir -p /tmp/initramfs/{bin,dev,proc,sys,root,tmp,etc}
cp /bin/busybox /tmp/initramfs/bin/
cd /tmp/initramfs
for cmd in sh mount umount ls cat echo mknod sleep dmesg cp mv rm \
    grep sed awk hexdump md5sum devmem ps kill top free vi tar \
    fdisk mkfs.ext2 mountpoint blkid ifconfig ping wget dmesg \
    modprobe sync reboot poweroff halt; do
    ln -sf /bin/busybox bin/$cmd
done
cp ~/tmac bin/
# ... create init script (see build_all.sh for full initramfs recipe) ...
find . | cpio -o -H newc | gzip > /tmp/initramfs.cpio.gz
```

---

## On Windows: Create BOOT.BIN and SD Card

### Prerequisites

- **Vivado 2023.1** installed (provides `bootgen` and builds `fsbl.elf`)
- The 5 boot files from macOS (copy to `D:\Users\u\tmac-zynq-fpga\linux\boot\`)
- **model.tmac** (~374 MB) — GGUF-converted model weights
- SD card (≥ 2 GB recommended)

### Step 1: Copy Files to Windows

Copy from macOS `~/arm-build/` to Windows `D:\Users\u\tmac-zynq-fpga\linux\boot\`:

```
linux/boot/
├── system_wrapper.bit     ← already in repo (FPGA bitstream from Vivado)
├── matmul_bd.xsa          ← already in repo (hardware handoff)
├── boot.bif               ← already in repo (bootgen config)
├── fsbl.elf               ← build in Vivado SDK (see below)
├── u-boot-spl.bin         ← copy from ~/arm-build/
├── u-boot.img             ← copy from ~/arm-build/
├── uImage                 ← copy from ~/arm-build/
├── devicetree.dtb          ← copy from ~/arm-build/
└── uramdisk.image.gz       ← copy from ~/arm-build/
```

### Step 2: Build FSBL in Vivado

In Vivado 2023.1:
1. Open hardware handoff: `File → Export → Export Hardware` from `matmul_bd.xsa`
2. Use the XSCT console:
```tcl
# In Vivado XSCT console:
hsi::open_hw_design linux/boot/matmul_bd.xsa
hsi::generate_app -hw linux/boot/matmul_bd.xsa -os standalone -proc ps7_cortexa9_0 -app zynq_fsbl
```
3. Copy the generated `fsbl.elf` to `linux/boot/`

### Step 3: Create BOOT.BIN

```cmd
cd D:\Users\u\tmac-zynq-fpga\linux\boot
bootgen -image boot.bif -o BOOT.BIN -w
```

`boot.bif` contents:
```
the_ROM_image:
{
    [bootloader] fsbl.elf
    system_wrapper.bit
    u-boot-spl.bin
}
```

### Step 4: Prepare SD Card

Format with two partitions:

| Partition | Type | Size | Contents |
|-----------|------|------|----------|
| 1 | FAT32 | 64 MB | BOOT.BIN, uImage, devicetree.dtb, uramdisk.image.gz |
| 2 | ext4 | Rest | model.tmac, tmac |

On Windows (using a tool like Rufus or diskpart for FAT32, and a Linux VM for ext4):

```
Partition 1 (FAT32):
    BOOT.BIN
    u-boot.img
    uImage
    devicetree.dtb
    uramdisk.image.gz

Partition 2 (ext4):
    model.tmac          (~374 MB — copy from models/)
    tmac                (483 KB — FPGA test program)
```

### Step 5: Boot the Board

1. **Power-cycle** the MicroPhase Z7-Lite (required — PLL re-init hangs on warm reset)
2. Insert SD card
3. Set boot mode DIP switches to SD card boot
4. **Connect UART** (115200 baud, 8N1) to monitor boot
5. Power on

### U-Boot Environment (auto-boot)

U-Boot loads `uImage` + `devicetree.dtb` from FAT32 partition 1 and boots.
Kernel auto-mounts the initramfs and runs the init script.

The initramfs init script:
1. Mounts `/proc`, `/sys`, `/dev`
2. Mounts SD ext4 partition to `/root`
3. The `tmac` binary runs from `/root/tmac`
4. Drops to BusyBox shell after tmac exits

**UART console** at 115200 baud. You'll see:
```
U-Boot 2022.01 ... (loading uImage)
Starting kernel ...
=== FPGA Linux Boot — Zynq 7010 ===
Ready. Commands:
  /root/tmac           — run FPGA test
  devmem 0x43C00014    — read REG_STATUS
  hexdump -C /dev/mem -s 0x43C00000 -n 64  — dump registers
```

### Debug: Manual U-Boot Boot

If auto-boot fails, stop at U-Boot prompt (press any key) and boot manually:

```
U-Boot> fatload mmc 0 0x3000000 uImage
U-Boot> fatload mmc 0 0x2A00000 devicetree.dtb
U-Boot> fatload mmc 0 0x2000000 uramdisk.image.gz
U-Boot> bootm 0x3000000 0x2000000 0x2A00000
```

### Debug: FPGA Register Access from Linux

Once booted into BusyBox shell:

```bash
# Read REG_STATUS (0x43C00014)
devmem 0x43C00014

# Read REG_DEBUG (0x43C00028) — FSM state + bus status
devmem 0x43C00028

# Dump all FPGA registers (64 bytes)
hexdump -C /dev/mem -s 0x43C00000 -n 64

# Run the full test program
/root/tmac
```

### Device Tree Note

The current `zynq-zc702.dtb` does NOT include the FPGA IP node.
Until a custom DTS with the `matmul@43c00000` node is added, the kernel
may restrict /dev/mem access to that range. Workaround: add
`iomem=relaxed` to the kernel command line (via U-Boot `bootargs`):

```
U-Boot> setenv bootargs "console=ttyPS0,115200 root=/dev/ram0 rw iomem=relaxed"
U-Boot> bootm 0x3000000 0x2000000 0x2A00000
```
