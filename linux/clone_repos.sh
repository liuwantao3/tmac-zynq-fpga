#!/bin/bash
# Clone all required repos for Linux-on-SD boot build
# Usage: bash linux/clone_repos.sh [workdir]
# Default workdir: /tmp/arm-build

set -euo pipefail

WORKDIR="${1:-/tmp/arm-build}"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

echo "=== Cloning Linux kernel (Xilinx, ~1.5GB shallow) ==="
git clone --depth=1 --single-branch --branch xilinx-v2024.1 \
    https://github.com/Xilinx/linux-xlnx.git "$WORKDIR/linux-xlnx" &
PID_KERNEL=$!

echo "=== Cloning U-Boot (Xilinx, ~200MB shallow) ==="
git clone --depth=1 --single-branch --branch xilinx-v2022.1 \
    https://github.com/Xilinx/u-boot-xlnx.git "$WORKDIR/u-boot-xlnx" &
PID_UBOOT=$!

echo "=== Cloning Buildroot (~100MB shallow) ==="
git clone --depth=1 \
    https://github.com/buildroot/buildroot.git "$WORKDIR/buildroot" &
PID_BR=$!

echo ""
echo "Downloads running in parallel (PIDs: kernel=$PID_KERNEL u-boot=$PID_UBOOT buildroot=$PID_BR)"
echo "Waiting for completion..."

wait $PID_KERNEL && echo "  ✓ linux-xlnx cloned" || echo "  ✗ linux-xlnx FAILED"
wait $PID_UBOOT  && echo "  ✓ u-boot-xlnx cloned"  || echo "  ✗ u-boot-xlnx FAILED"
wait $PID_BR     && echo "  ✓ buildroot cloned"    || echo "  ✗ buildroot FAILED"

echo ""
echo "=== Repo sizes ==="
du -sh "$WORKDIR"/linux-xlnx "$WORKDIR"/u-boot-xlnx "$WORKDIR"/buildroot 2>/dev/null

echo ""
echo "All repos in: $WORKDIR"
echo "Next: run linux/build_all.sh"
