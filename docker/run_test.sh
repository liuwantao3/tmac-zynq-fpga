#!/bin/bash
# Run T-MAC inference test in Docker (x86_64)
# Tests the same firmware logic as ARM, just compiled for host

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "T-MAC Inference Test (QEMU Simulation)"
echo "=========================================="

# Build Docker image if not exists
IMAGE_NAME="tmac-inference"

if ! docker image inspect $IMAGE_NAME &>/dev/null; then
    echo "Building Docker image..."
    docker build -t $IMAGE_NAME -f Dockerfile.firmware ..
else
    echo "Using cached Docker image"
fi

# Run the test
echo ""
echo "Running inference test..."
docker run --rm \
    -v $(pwd)/../models:/workspace/models \
    $IMAGE_NAME \
    bash -c "cd /workspace/sim && g++ -O2 -o tmac_test tmac_test.cpp -lm && ./tmac_test"

echo ""
echo "=========================================="
echo "Test complete!"
echo "=========================================="
