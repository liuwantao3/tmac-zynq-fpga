#!/bin/bash
# Build U-Boot + Linux kernel + Buildroot for Zynq-7010 SD card boot
# Requires: step 1 (clone) completed, arm-linux-gnueabihf- toolchain in PATH
# Usage: bash linux/build_all.sh [workdir] [fpga_root]

set -euo pipefail

WORKDIR="${1:-/tmp/arm-build}"
FPGA_ROOT="${2:-$(cd "$(dirname "$0")/.." && pwd)}"
BOOT_DIR="$FPGA_ROOT/linux/boot"
CORES=$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)

echo "=== Build environment ==="
echo "  workdir:   $WORKDIR"
echo "  fpga_root: $FPGA_ROOT"
echo "  boot_dir:  $BOOT_DIR"
echo "  cores:     $CORES"
echo ""

which arm-linux-gnueabihf-gcc >/dev/null 2>&1 || {
    echo "ERROR: arm-linux-gnueabihf-gcc not found in PATH."
    echo "Install ARM GCC from: https://developer.arm.com/downloads/-/arm-gnu-toolchain-downloads"
    echo "Or set CROSS_COMPILE manually and rerun."
    exit 1
}
GCC_VER=$(arm-linux-gnueabihf-gcc --version | head -1)
echo "  compiler:  $GCC_VER"
echo ""

export CROSS_COMPILE=arm-linux-gnueabihf-
TOOLCHAIN_DIR=$(dirname "$(which arm-linux-gnueabihf-gcc)")
export PATH="$TOOLCHAIN_DIR:$PATH"

# ── 1. U-Boot ──
echo "=== [1/3] Building U-Boot ==="
cd "$WORKDIR/u-boot-xlnx"
make zynq_zc702_defconfig
make -j"$CORES"
cp u-boot "$BOOT_DIR/u-boot.elf"
echo "  → u-boot.elf copied to $BOOT_DIR/"
echo ""

# ── 2. Linux Kernel ──
echo "=== [2/3] Building Linux Kernel ==="
cd "$WORKDIR/linux-xlnx"
make ARCH=arm xilinx_zynq_defconfig
make -j"$CORES" ARCH=arm UIMAGE_LOADADDR=0x8000 uImage
make ARCH=arm dtbs
cp arch/arm/boot/uImage "$BOOT_DIR/"
cp arch/arm/boot/dts/zynq-zc702.dtb "$BOOT_DIR/devicetree.dtb"
echo "  → uImage, devicetree.dtb copied to $BOOT_DIR/"
echo ""

# ── 3. Buildroot (rootfs) ──
echo "=== [3/3] Building Buildroot rootfs ==="
cd "$WORKDIR/buildroot"
make qemu_arm_vexpress_defconfig

# Enable NEON/VFPv3 for Cortex-A9
cat >> .config << 'BRCFG'
BR2_ARM_ENABLE_VFP=y
BR2_ARM_ENABLE_NEON=y
BRCFG
make olddefconfig
make -j"$CORES"

cp output/images/rootfs.cpio.uboot "$BOOT_DIR/uramdisk.image.gz"
echo "  → uramdisk.image.gz copied to $BOOT_DIR/"
echo ""

# ── Summary ──
echo "============================================"
echo "  Build complete. Boot files in $BOOT_DIR/:"
echo "============================================"
ls -lh "$BOOT_DIR"/{u-boot.elf,uImage,devicetree.dtb,uramdisk.image.gz} 2>/dev/null
echo ""
echo "Next steps:"
echo "  1. On Windows with Vivado: cd $BOOT_DIR && bootgen -image boot.bif -o BOOT.BIN -w"
echo "  2. Format SD card: partition 1 FAT32, partition 2 ext4"
echo "  3. Copy BOOT.BIN + uImage + devicetree.dtb + uramdisk.image.gz → FAT32 partition"
echo "  4. Copy model.tmac + tmac → ext4 partition"
echo "  5. Insert SD, set boot mode to SD, power on"
