# Documentation

## Documents
- [architecture.md](architecture.md) — Full system architecture: model dimensions, quantization types, dequant formulas, bugs found, inference pipeline
- [AGENTS.md](AGENTS.md) — FPGA development workflow (HLS kernel + three-layer feedback system)
- [Q4_K_IMPLEMENTATION_PLAN.md](Q4_K_IMPLEMENTATION_PLAN.md) — Q4_K implementation plan (archived post-implementation)
- [PROGRESS_SUMMARY.md](PROGRESS_SUMMARY.md) — Historical progress tracking (INT4 era, kept for reference)
- [FPGA_PERFORMANCE_ANALYSIS.md](FPGA_PERFORMANCE_ANALYSIS.md) — Theoretical Zynq 7010 performance analysis (Q8_0 baseline)
- [hls_q8_kernel_explanation.md](hls_q8_kernel_explanation.md) — Legacy HLS Q8 kernel line-by-line explanation

## Project Context
- **Model**: Qwen2-0.5B-Instruct (GGUF q4_k_m quantization)
- **Inference**: C++ simulation (`sim/tmac_gguf.cpp`) + Verilog RTL accelerator (`verilog/matmul_top.v`)
- **Format**: TMAC (converted from GGUF via `scripts/extract_tmac.py`)
- **Target**: Zynq 7010 (512MB DDR) — dual-core Verilog accelerator (Q8_0 + Q4_K)

## Directory Map
| Directory | Purpose |
|-----------|---------|
| `sim/` | C++ inference engine + FPGA simulation |
| `verilog/` | Verilog RTL accelerator (primary implementation) |
| `scripts/` | Python ground truth, verification, conversion tools |
| `hls/` | HLS kernel source (legacy) |
| `firmware/` | ARM runtime skeleton |
| `vivado/` | Vivado block design scripts |
| `models/` | GGUF source model file |
