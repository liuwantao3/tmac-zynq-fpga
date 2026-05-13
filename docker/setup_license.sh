#!/bin/bash
# Xilinx License Setup Script
# Handles license file mounting and configuration

set -e

LICENSE_DIR="${1:-$(pwd)/licenses}"
CONTAINER_LICENSE_PATH="/workspace/xilinx.lic"

usage() {
    echo "Xilinx License Setup"
    echo ""
    echo "Usage: $0 <license_directory>"
    echo ""
    echo "Place your Xilinx .lic file in the license directory."
    echo "The license will be mounted to /workspace/xilinx.lic in container."
    echo ""
    echo "License Environment Variables:"
    echo "  XILINX_LICENSE_FILE - Points to license file"
    echo ""
}

check_license() {
    if [ ! -f "${LICENSE_DIR}/xilinx.lic" ]; then
        echo "ERROR: License file not found at ${LICENSE_DIR}/xilinx.lic"
        echo ""
        echo "Please place your Xilinx license file there."
        echo "You can get a license from: https://www.xilinx.com/getlicense"
        return 1
    fi

    echo "Found license: ${LICENSE_DIR}/xilinx.lic"
    return 0
}

validate_license() {
    echo "Validating license..."
    if grep -q "XILINX_LICENSE_FILE" "${LICENSE_DIR}/xilinx.lic" 2>/dev/null || \
       grep -q "SERVER" "${LICENSE_DIR}/xilinx.lic" 2>/dev/null; then
        echo "License file appears valid"
        return 0
    else
        echo "Warning: License file may be invalid"
        return 1
    fi
}

setup_for_docker() {
    mkdir -p "${LICENSE_DIR}"
    echo ""
    echo "For Docker Compose, add to environment:"
    echo "  XILINX_LICENSE_FILE=${CONTAINER_LICENSE_PATH}"
    echo ""
    echo "Mount in docker-compose.yml:"
    echo "  - ${LICENSE_DIR}:/workspace/licenses:ro"
    echo ""
}

main() {
    mkdir -p "${LICENSE_DIR}"

    echo "=== Xilinx License Setup ==="
    echo ""

    if check_license; then
        validate_license
    else
        echo ""
        echo "To use Xilinx tools, you need a license."
        echo ""
        echo "Options:"
        echo "1. Free license: https://www.xilinx.com/products/silinx-ip.html"
        echo "2. Evaluation license: https://www.xilinx.com/products/silicon-devices.html"
        echo "3. University program: https://www.xilinx.com/support/university.html"
        echo ""
    fi

    setup_for_docker
}

main "$@"