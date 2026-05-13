#!/bin/bash
# FPGA Docker Setup Script
# Usage: ./docker/setup.sh [build|start|shell|clean]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(dirname "${SCRIPT_DIR}")"
DOCKER_DIR="${SCRIPT_DIR}"
XILINX_VERSION="2023.1"

usage() {
    echo "FPGA Docker Setup"
    echo ""
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  download  - Download Xilinx installers"
    echo "  build     - Build Docker image"
    echo "  start     - Start container"
    echo "  shell     - Attach to running container"
    echo "  stop      - Stop container"
    echo "  clean     - Clean Docker resources"
    echo ""
}

download_xilinx() {
    echo "=== Downloading Xilinx Tools ==="
    echo "WARNING: This downloads ~60GB"
    echo ""

    local install_dir="${DOCKER_DIR}/xilinx_installers"
    mkdir -p "${install_dir}"

    local vivado_url="https://www.xilinx.com/member/forms/download/vivado.html?filename=Vivado_${XILINX_VERSION}_preliminary.tar.gz"
    local vitis_url="https://www.xilinx.com/member/forms/download/vitis.html?filename=Vitis_HLS_${XILINX_VERSION}.tar.gz"

    echo "Download locations:"
    echo "  Vivado: ${vivado_url}"
    echo "  Vitis HLS: ${vitis_url}"
    echo ""
    echo "Save to: ${install_dir}/"
    echo ""
    read -p "Download now? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Download skipped. Place installers manually in ${install_dir}/"
        return 1
    fi

    echo "Note: Direct download links require Xilinx login."
    echo "Please download manually from:"
    echo "  https://www.xilinx.com/download.html"
    echo ""
}

build_image() {
    echo "=== Building FPGA Docker Image ==="

    if [ ! -d "${DOCKER_DIR}/xilinx_installers" ] || [ -z "$(ls -A ${DOCKER_DIR}/xilinx_installers 2>/dev/null)" ]; then
        echo "ERROR: No Xilinx installers found in ${DOCKER_DIR}/xilinx_installers/"
        echo "Run: $0 download"
        exit 1
    fi

    cd "${DOCKER_DIR}"

    echo "Building fpga-builder image..."
    docker build -t fpga-builder:latest -f Dockerfile .

    echo ""
    echo "Building fpga-build-cpu image..."
    docker build -t fpga-build-cpu:latest -f Dockerfile.cpu .

    echo ""
    echo "=== Build Complete ==="
    echo "Run: $0 start"
}

start_container() {
    echo "=== Starting FPGA Container ==="

    if ! docker info > /dev/null 2>&1; then
        echo "ERROR: Docker is not running"
        exit 1
    fi

    cd "${WORKSPACE}"

    docker-compose up -d
    echo ""
    echo "Container started. Run: docker-compose exec fpga-builder bash"
}

attach_shell() {
    echo "=== Attaching to FPGA Container ==="
    docker-compose exec fpga-builder bash
}

stop_container() {
    echo "=== Stopping FPGA Container ==="
    docker-compose down
}

clean_resources() {
    echo "=== Cleaning Docker Resources ==="
    read -p "Remove Docker images? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Clean cancelled"
        return
    fi

    docker rmi fpga-builder:latest fpga-build-cpu:latest 2>/dev/null || true
    docker-compose down -v 2>/dev/null || true
    echo "Clean complete"
}

main() {
    local cmd="${1:-usage}"

    case "${cmd}" in
        download)
            download_xilinx
            ;;
        build)
            build_image
            ;;
        start)
            start_container
            ;;
        shell)
            attach_shell
            ;;
        stop)
            stop_container
            ;;
        clean)
            clean_resources
            ;;
        *)
            usage
            ;;
    esac
}

main "$@"