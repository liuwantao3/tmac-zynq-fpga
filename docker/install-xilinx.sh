#!/bin/bash
# Xilinx Installation Script for Docker + Rosetta
# Run from fpga project root: ./docker/install-xilinx.sh

set -e

echo "================================================"
echo "Xilinx Installation via Docker + Rosetta"
echo "================================================"

# Configuration
XILINX_INSTALL_DIR="/opt/Xilinx"
XILINX_VERSION="2023.1"
IMAGE_NAME="fpga-xilinx-installer"
CONTAINER_NAME="xilinx-installer"

# Check if external drive is accessible
if [ ! -d "/Volumes/Xilinx/Xilinx_Unified_2023.1_0507_1903" ]; then
    echo "ERROR: Xilinx installer not found at /Volumes/Xilinx/"
    exit 1
fi

echo "Step 1: Building base Docker image..."
cd /Users/arctic/fpga
docker build -f docker/Dockerfile.xilinx-base -t ${IMAGE_NAME}:base .

echo "Step 2: Starting installation container..."
# This runs the xsetup installer with volume mount
docker rm -f ${CONTAINER_NAME} 2>/dev/null || true

docker run -d --name ${CONTAINER_NAME} \
    --platform=linux/amd64 \
    -v /Volumes/Xilinx:/xilinx:ro \
    -v $(pwd):/workspace \
    -w /workspace \
    ${IMAGE_NAME}:base \
    tail -f /dev/null

echo "Step 3: Checking installer files..."
docker exec ${CONTAINER_NAME} ls -la /xilinx/Xilinx_Unified_2023.1_0507_1903/xsetup

echo "Step 4: Checking payload status..."
docker exec ${CONTAINER_NAME} ls /xilinx/Xilinx_Unified_2023.1_0507_1903/payload/ | wc -l

echo "================================================"
echo "Container is running. To proceed with installation:"
echo ""
echo "  docker exec -it ${CONTAINER_NAME} bash"
echo ""
echo "Then run:"
echo "  cd /xilinx/Xilinx_Unified_2023.1_0507_1903"
echo "  ./xsetup"
echo ""
echo "Or to run batch installation:"
echo "  yes | ./xsetup --batch"
echo "================================================"