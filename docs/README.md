# Documentation

## Key Documents
- [architecture.md](architecture.md) — Full system architecture: model dimensions, quantization types, dequant formulas, bugs found, inference pipeline
- [AGENTS.md](AGENTS.md) — FPGA development workflow and HLS kernel documentation
- [PROGRESS_SUMMARY.md](PROGRESS_SUMMARY.md) — Historical progress tracking

## Project Context
- **Model**: Qwen2-0.5B-Instruct (GGUF q4_k_m quantization)
- **Inference**: Standalone C++ (`sim/tmac_gguf.cpp`), verified against gguf Python library
- **Format**: TMAC (converted from GGUF via `scripts/extract_tmac.py`)
- **Target**: Zynq 7010 (512MB DDR) — FPGA design aspirational

## Directory Map
| Directory | Purpose |
|-----------|---------|
| `sim/` | C++ inference engine (working) |
| `scripts/` | Python ground truth, verification, conversion tools |
| `hls/` | HLS kernel source (aspirational) |
| `firmware/` | ARM runtime (aspirational) |
| `vivado/` | Vivado block design (aspirational) |
| `models/` | GGUF source model file |
