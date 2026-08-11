#!/bin/bash
# Clone the Linux kernel + U-Boot sources needed to rebuild the Z7-Lite
# SD-boot artifacts inside WSL (see linux/build_wsl.sh). No buildroot — the
# initramfs is assembled from a busybox source build + gen_init_cpio.
# Usage: bash linux/clone_repos.sh [workdir]
# Default workdir: /home/u  (the WSL user home)

set -euo pipefail

WORKDIR="${1:-/home/u}"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

echo "=== Cloning Linux kernel (Xilinx, shallow) ==="
if [ ! -d linux-xlnx ]; then
    git clone --depth=1 --single-branch --branch xilinx-v2024.1 \
        https://github.com/Xilinx/linux-xlnx.git "$WORKDIR/linux-xlnx"
else
    echo "  linux-xlnx already present"
fi

echo "=== Cloning U-Boot (Xilinx, shallow — only for its tools/mkimage + dts) ==="
if [ ! -d u-boot-xlnx ]; then
    git clone --depth=1 --single-branch --branch xilinx-v2022.1 \
        https://github.com/Xilinx/u-boot-xlnx.git "$WORKDIR/u-boot-xlnx"
else
    echo "  u-boot-xlnx already present"
fi

echo ""
echo "=== Package deps (Ubuntu 24.04 WSL) ==="
echo "  sudo apt install gcc-arm-linux-gnueabihf build-essential bison flex bc"
echo "    libssl-dev libelf-dev u-boot-tools device-tree-compiler cpio wget"
echo ""
echo "All sources in: $WORKDIR"
echo "Next: bash linux/build_wsl.sh"
