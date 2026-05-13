# T-MAC FPGA Development Environment

Docker-based Xilinx Vivado / Vitis HLS development environment.

## Quick Start
```bash
cd /Users/arctic/fpga/docker
./setup.sh build   # Build Docker image with Vivado
./setup.sh start   # Start container
./setup.sh shell   # Attach shell
```

## File Layout
```
docker/
├── Dockerfile               # Full Vivado + Vitis HLS image
├── Dockerfile.cpu           # CPU-only (ARM firmware compilation)
├── Dockerfile.firmware      # Lightweight firmware build
├── Dockerfile.inference     # Inference-only environment
├── Dockerfile.arm64         # ARM64 cross-compilation
├── Dockerfile.qemu          # QEMU emulation
├── docker-compose.yml       # Container configuration
├── setup.sh                 # Build/start scripts
├── setup_license.sh         # Xilinx license setup
├── setup_xilinx.sh          # Xilinx installer setup
├── run_test.sh              # Test runner
├── licenses/                # License files
└── xilinx_installers/       # Vivado installers (119GB tar.gz)
```

## Notes
- Vivado installation requires ~60-80 GB disk space and 1-3 hours
- For software-only work, use `Dockerfile.cpu` or `Dockerfile.firmware` (no Vivado needed)
- The verified C++ inference engine (`sim/tmac_gguf.cpp`) runs natively on macOS/Linux
