#!/bin/bash
# Xilinx Installer Setup Script for Docker

set -e

XILINX_DIR=/opt/Xilinx
INSTALLER_DIR=${XILINX_DIR}/Xilinx_Unified_2023.1_0507_1903

echo "=== Xilinx Docker Setup ==="

# Create directories
mkdir -p ${XILINX_DIR}
mkdir -p /root/.Xilinx
mkdir -p /workspace

# Copy license
cp /tmp/xilinx.lic /root/.Xilinx/Xilinx.lic 2>/dev/null || true

# Extract installer if not already extracted
if [ ! -d "${INSTALLER_DIR}" ]; then
    echo "Extracting Xilinx Unified installer..."
    tar -xzf /tmp/Xilinx.tar.gz -C ${XILINX_DIR}/
fi

echo "Checking installer at ${INSTALLER_DIR}..."
ls -la ${INSTALLER_DIR}/ 2>/dev/null | head -10

# Make xsetup executable
chmod +x ${INSTALLER_DIR}/xsetup 2>/dev/null || true

echo "=== Installation files ready ==="
echo "XILINX_DIR: ${XILINX_DIR}"
echo "INSTALLER: ${INSTALLER_DIR}"

# Check if we can run xsetup in batch mode
if [ -f "${INSTALLER_DIR}/bin/xsetup" ]; then
    chmod +x ${INSTALLER_DIR}/bin/xsetup
    echo "xsetup found at ${INSTALLER_DIR}/bin/xsetup"
fi

echo "Done."