# Documentation

## Documents
- [architecture.md](architecture.md) — Full system architecture: model dimensions, quantization types, dequant formulas, bugs found, inference pipeline
- [AGENTS.md](AGENTS.md) — FPGA development workflow + full architecture, register map, debug guide
- [infrastructure.md](infrastructure.md) — Board infrastructure & lessons learned: UART0 console, DDR3/AXI HP0, PS7 clocks/PLL, JTAG/DAP
- [debug_experiences.md](debug_experiences.md) — Debugging toolbox: ILA (Integrated Logic Analyzer) implementation + runtime + analysis (first ILA campaign 2026-08-16), iVerilog sim discipline, on-board compare/trace, golden models, host-side repros
- [debug_procedures.md](debug_procedures.md) — XSDB/DAP/ps7_init quick reference, AFI regs, FSM-hang decode
- [q5_q8_optimization_plan.md](q5_q8_optimization_plan.md) — Q5/Q8 optimization plan (throughput-first): phases, levers, sequencing
- [q5_q8_core_analysis.md](q5_q8_core_analysis.md) — Line-by-line walkthrough of the Q5_0/Q8_0 cores + FSM interaction, with a cycle-level model
- [Q4_K_IMPLEMENTATION_PLAN.md](Q4_K_IMPLEMENTATION_PLAN.md) — Q4_K implementation plan (archived post-implementation)
- [PROGRESS_SUMMARY.md](PROGRESS_SUMMARY.md) — Historical progress tracking (INT4 era, kept for reference)
- [hls_q8_kernel_explanation.md](hls_q8_kernel_explanation.md) — Legacy HLS Q8 kernel line-by-line explanation

## Project Context
- **Model**: Qwen2-0.5B-Instruct (GGUF q4_k_m quantization)
- **Inference**: C++ simulation (`sim/tmac_gguf.cpp`) + Verilog RTL accelerator (`hp_fsm_top.v` / `matmul_top.v`)
- **Format**: TMAC (converted from GGUF via `scripts/extract_tmac.py`)
- **Target**: Zynq 7010 (512MB DDR) — multi-core Verilog accelerator (Q8_0, Q5_0, Q4_K, Q6_K, INT16)

## Directory Map
| Directory | Purpose |
|-----------|---------|
| `sim/` | C++ inference engine + FPGA simulation |
| `verilog/` | Verilog RTL accelerator (primary implementation) + testbenches |
| `scripts/` | Python ground truth, verification, conversion tools |
| `vivado_integration/` | Active Vivado/Vitis integration: build script, bare-metal ARM port |
| `linux/` | Linux-on-SD build guide + boot files + JTAG bring-up scripts (WSL build, verified SD boot) |
| `models/` | GGUF source + TMAC model files |
| `hls/` | HLS kernel source (legacy, archived) |
| `firmware/` | ARM runtime skeleton (aspirational, archived) |
| `descriptor-orchestrator/` | Early descriptor orchestrator prototype (archived) |
| `vivado/` | HLS-era Vivado block design stub (archived) |
