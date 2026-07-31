#!/bin/bash
# Build Xilinx bootgen (inside the Lima VM) and fuse BOOT.BIN from
# fsbl.elf + system_wrapper.bit + u-boot.elf. Runs entirely on the Mac side —
# no Windows/Vivado needed to make the SD boot image.
#
# The committed linux/boot/fsbl.elf is built from matmul_bd.xsa and is only
# regenerable on Windows (XSCT: hsi::generate_app -app zynq_fsbl) — it only
# needs to change if the hardware (bitstream/XSA) changes.
#
# bootgen is built from the buildroot tree via `make host-bootgen`
# (output/host/bin/bootgen), falling back to a direct clone of Xilinx/bootgen.
# Usage: bash linux/build_bootbin.sh [workdir] [fpga_root]

set -euo pipefail

WORKDIR="${1:-/tmp/arm-build}"
FPGA_ROOT="${2:-$(cd "$(dirname "$0")/.." && pwd)}"
BOOT_DIR="$FPGA_ROOT/linux/boot"

echo "=== Building BOOT.BIN on the Mac (Lima VM) ==="
echo "  workdir:   $WORKDIR"
echo "  boot_dir:  $BOOT_DIR"
echo ""

# ── 1. Locate / build bootgen ──
BOOTGEN="$WORKDIR/buildroot/output/host/bin/bootgen"
if [ ! -x "$BOOTGEN" ]; then
    if [ -d "$WORKDIR/buildroot" ]; then
        echo "Building bootgen via buildroot host-bootgen..."
        make -C "$WORKDIR/buildroot" host-bootgen
    fi
fi
if [ ! -x "$BOOTGEN" ]; then
    echo "Building bootgen from Xilinx/bootgen source..."
    git clone --depth=1 https://github.com/Xilinx/bootgen.git "$WORKDIR/bootgen-src"
    make -C "$WORKDIR/bootgen-src" \
        LIBS="$(pkg-config --libs libssl libcrypto)" \
        INCLUDE_USER="$(pkg-config --cflags libssl libcrypto)"
    BOOTGEN="$WORKDIR/bootgen-src/build/bin/bootgen"
fi
[ -x "$BOOTGEN" ] || { echo "ERROR: bootgen not found/built."; exit 1; }
echo "  bootgen:   $BOOTGEN"

# ── 2. Inputs must all be present ──
for f in fsbl.elf system_wrapper.bit u-boot.elf boot.bif; do
    [ -f "$BOOT_DIR/$f" ] || { echo "ERROR: $BOOT_DIR/$f missing — run linux/build_all.sh first."; exit 1; }
done

# ── 3. Create BOOT.BIN (relative paths in boot.bif resolve from BOOT_DIR) ──
cd "$BOOT_DIR"
"$BOOTGEN" -image boot.bif -o BOOT.BIN -w
echo ""
echo "  → BOOT.BIN created: $(ls -la BOOT.BIN | awk '{print $5" bytes"}')"
echo "    Copy BOOT.BIN + uImage + devicetree.dtb + uramdisk.image.gz + boot.scr"
echo "    to the SD FAT32 p1, model.tmac + tmac to p2, J1=SD, power-cycle."
