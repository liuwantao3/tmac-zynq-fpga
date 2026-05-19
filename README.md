# T-MAC Zynq 7010 GGUF Inference

Standalone C++ inference engine for **Qwen2-0.5B-Instruct** on a **Xilinx Zynq 7010** (512MB DDR),
with a Q8_0 direct-path FPGA accelerator (HLS, not yet synthesized).

No llama.cpp dependency — reads weights via GGUF→TMAC conversion.

---

## 1. Project Status

| Component | Status | Verified |
|-----------|--------|----------|
| C++ inference engine (`sim/tmac_gguf.cpp`) | **Complete** | FP32 matches ground truth across all 24 layers (max diff < 0.0003), logits (max diff < 0.0003) |
| GGUF → TMAC converter (`scripts/extract_tmac.py`) | **Complete** | Output `/tmp/model.tmac` (373.7 MB, 290 tensors) |
| FPGA simulation (`sim/fpga_sim.hpp`) | **Complete** | Cycle-accurate MatmulAccel with AXI-Lite register model + interrupt |
| Verilog Q8_0 accelerator (`verilog/`) | **Complete** | 6/6 tests pass, 5 cosim tiles (320 checks) pass |
| HLS kernel Q8_0 direct path (`hls/matmul_q8.cpp`) | **Deprecated** | Superseded by Verilog RTL |
| ARM firmware (`firmware/`) | **Aspirational** | Stub — needs full implementation |
| Vivado block design (`vivado/block_design.tcl`) | **Aspirational** | Stub — needs HLS/Verilog export first |

---

## 2. Hardware Target: Zynq 7010 (xc7z010clg400-1)

| Resource | Available | Used (Verilog) | Notes |
|----------|-----------|---------------|-------|
| DSP slices | 80 | 8 (10%) | 8× INT16 MAC (one per lane) |
| LUT | 17,600 | ~2.5-3.5K (est.) | Post-synthesis required for exact |
| BRAM | 135 KB (60 × BRAM18K) | 1 BRAM36 (36 KB) | Weight storage: 512×64-bit |
| FF | 35,200 | ~6.5-9.7K (est.) | Accumulators + pipeline regs |
| DDR3 | 512 MB | ~374 MB | Model weights |

---

## 3. Model: Qwen2-0.5B-Instruct

| Parameter | Value |
|-----------|-------|
| Hidden dim | 896 |
| FFN intermediate dim | 4864 |
| Vocab size | 151,936 |
| Layers | 24 |
| Attention heads | 14 |
| Head dim | 64 |
| KV heads (GQA) | 2 |
| RoPE theta | 1,000,000 |
| RMS norm eps | 1e-6 |
| Tie embeddings | Yes (lm_head = token_embd.weight) |

### Quantization Type Distribution

| Type | Tensors | Where Used |
|------|---------|------------|
| Q5_0 | 132 | Most weight matrices (good accuracy, 5.21 bpw) |
| Q6_K | 12 | Selected `blk.*.ffn_down.weight` (higher precision for FFN output) |
| Q8_0 | 2 | `token_embd.weight`, `attn_v.weight` (critical paths, full byte) |
| Q4_K | rest | Remaining weights |
| F32 | few | Norm weights, biases (negligible size) |

---

## 4. Repository Structure

```
/Users/arctic/fpga/
├── README.md                  ← This file — project overview & handover
├── .gitignore
├── Makefile                   ← FPGA build targets (HLS, Vivado, clean)
│
├── sim/                       ← C++ host simulation (MATURE)
│   ├── tmac_gguf.cpp          ← Main inference engine (1299 lines)
│   ├── matmul_q8.cpp          ← Q8_0 direct-path matmul (simulated FPGA dequant)
│   ├── fpga_sim.hpp           ← FPGA simulator: MatmulAccel, AXI-Lite, timing model
│   ├── chat.py                ← Chat interface (Python, uses tmac_gguf binary)
│   ├── vocab.json             ← Qwen2 tokenizer vocab
│   ├── merges.txt             ← BPE merges
│   └── Transaction Tracer/    ← AXI transaction tracing utilities
│
├── verilog/                  ← Verilog RTL accelerator (PRIMARY — COMPLETE)
│   ├── matmul_q8_core.v      ← 3-stage pipeline Q8_0 compute core
│   ├── dequant_lut.v         ← Q8_0 dequant LUT (standalone)
│   ├── systolic_8x8.v       ← 8×8 systolic array (standalone, not used)
│   ├── matmul_q8_top.v      ← Top-level: AXI4-Lite + BRAM buffers
│   ├── axilite_slave.v      ← AXI4-Lite slave + register file
│   ├── tb_matmul_q8.v        ← Core testbench (6 tests, all pass)
│   ├── tb_cosim.v           ← Cosimulation with real model tiles
│   ├── Makefile             ← iverilog build targets
│   └── DESIGN.md            ← Verilog design document

├── hls/                       ← HLS kernel sources (DEPRECATED — superseded by Verilog)
│   ├── matmul_q8.cpp          ← PRIMARY: Q8_0 direct path, LUT scale mult, INT16 systolic
│   ├── matmul_q8.hpp          ← Type/constant defines for matmul_q8
│   ├── matmul_int16.cpp       ← LEGACY: INT16×INT16→INT64, same 8×8 systolic
│   ├── matmul_int16.hpp       ← Type defines for INT16 kernel
│   ├── matmul_int8.cpp        ← DEPRECATED: INT8×INT8→INT32 (insufficient precision)
│   ├── script_q8.tcl          ← HLS synthesis script for matmul_q8 (recommended)
│   ├── script_int16.tcl       ← HLS script for INT16 fallback
│   ├── script.tcl             ← HLS script for INT8 (deprecated)
│   ├── hls_config.tcl         ← Shared HLS directives (config_bind -mul_style luts)
│   └── README.md              ← HLS-specific notes
│
├── firmware/                  ← ARM runtime (ASPIRATIONAL — stubs need rewrite)
│   ├── tmac_fpga.hpp/.cpp     ← FPGA driver interface (updated for matmul_q8)
│   ├── tmac_runtime.hpp/.cpp  ← Core inference runtime on ARM
│   ├── tmac_app.cpp           ← Main entry point for ARM
│   └── README.md              ← Firmware design notes
│
├── vivado/                    ← Vivado integration (ASPIRATIONAL)
│   └── block_design.tcl       ← Block design script (needs HLS IP export first)
│
├── scripts/                   ← Python utilities & verification
│   ├── extract_tmac.py        ← GGUF → TMAC binary converter
│   ├── ground_truth_v2.py     ← Ground truth via gguf Python library
│   ├── gguf_inference.py      ← FP32 inference via GGUFReader
│   ├── gguf_layer_inference.py ← Vectorized layer-by-layer ground truth
│   ├── py_tmac_vec.py         ← Vectorized Python TMAC reference (matches C++ exactly)
│   ├── verify_layers_fast.py  ← Compare C++ vs Python layer by layer
│   ├── compare_weights.py     ← Dequantization element-by-element comparison
│   ├── llama_dump.c           ← Reference patch for llama.cpp ground truth
│   ├── feedback_parser.py     ← HLS/Vivado report parser (three-layer feedback)
│   └── design_iteration.sh    ← FPGA design iteration loop
│
├── models/                    ← Model weight files (large, gitignored)
│   ├── qwen2-0_5b-instruct-q4_k_m.gguf  ← Source GGUF (~392 MB)
│   ├── model.tmac                        ← Generated TMAC (~374 MB)
│   └── README.md                         ← Model download/conversion notes
│
├── docs/                      ← Documentation
│   ├── architecture.md        ← Full architecture, quantization formats, bug history
│   ├── AGENTS.md              ← FPGA design workflow for AI agents (Vivado/HLS commands)
│   └── PROGRESS_SUMMARY.md    ← Historical progress (INT4 era, kept for reference)
│
├── licenses/                  ← Xilinx license file (gitignored)
│   └── xilinx.lic
│
├── learning/                  ← Historical exploration & prototypes
│   ├── py_tmac_inference.py   ← Early Python inference prototype
│   ├── py_tmac_vec.py         ← Vectorized early reference
│   ├── tmac_gguf.cpp          ← Early C++ prototype
│   ├── tmac_gguf_before_comp_llama.cpp
│   ├── py_tmac_inference_before_comp_llama.py
│   ├── debug_process.md       ← Debug methodology notes
│   └── SESSION_SUMMARY_May_8.md
│
└── gguf-tools-main/           ← Third-party GGUF inspection tool
    ├── gguf-tools.c / gguflib.c / sds.c / fp16.c
    ├── Makefile
    └── README.md
```

---

## 5. Key Architecture Decisions

### 5.1 INT4 Abandoned → Q5_0/Q8_0 Mixed Quantization

The project started with a custom INT4 format (block size 4). Deep layers (7+) showed multiplicative
error accumulation through SwiGLU residual connections, causing value explosion and NaN by layer 17.

**Switch to GGUF-sourced weights**: Replaced custom INT4 with llama.cpp's proven quantization formats.
The GGUF model uses Q5_0 (5-bit) for most weights, Q6_K for FFN output layers, and Q8_0 for
critical tensors (embedding, V projection). Block size increased from 4→32/256 elements, giving
significantly better statistical accuracy.

### 5.2 INT8 Insufficient → INT16 Direct MAC

The Verilog accelerator uses 8 parallel INT16 MAC lanes (not systolic). INT8×INT8 (8-bit)
causes catastrophic SwiGLU divergence by layer 3 (max logit diff 35.6, top-1 wrong). INT16×INT16
achieves near-identical results to FP32 (max logit diff 0.24, top-5 match 5/5). Zynq 7010
DSP48E1 handles INT16×INT16 in one slice (no extra resource cost vs INT8).

### 5.3 Q8_0 Direct Path: Dequant on FPGA, Not ARM

**Approach**: FPGA receives raw Q8_0 weight bytes + precomputed UQ8.8 combined scales.
FPGA handles Q8→INT16 dequantization via combinational multipliers (LUT-based, 0 DSP),
then INT16×INT16 direct MAC with 8 parallel lanes (8 DSPs).

**Tile compute**: 515 cycles/tile @ 150 MHz → 120,596 tiles → ~413 ms/token.

### 5.4 GGUF → TMAC Format

TMAC is a flat binary format that mirrors GGUF tensor data 1:1 (same bit patterns).
Purpose: simplify loading on bare-metal ARM (no GGUF parser needed). The converter
(`scripts/extract_tmac.py`) strips GGUF metadata headers and writes raw tensor blobs
with minimal headers (name, shape, type, size).

### 5.5 Column-Major Storage Convention

GGUF stores 2D tensors column-major: `ne[0]` = input_dim (fast), `ne[1]` = output_dim (slow).
The C++ engine accesses correctly: `idx = col + row * input_dim`. Square matrices (896×896)
are symmetric so row vs column order doesn't matter, but non-square projections (K:128×896,
V:128×896, FFN:4864×896) fail silently if accessed row-major.

---

## 6. C++ Inference Engine (`sim/tmac_gguf.cpp`)

### 6.1 Build

```bash
cd sim
# Release build:
g++ -std=c++17 -pthread -O2 -o tmac_gguf tmac_gguf.cpp matmul_q8.cpp

# Debug / ASAN:
g++ -std=c++17 -pthread -O0 -g -o tmac_gguf_dbg tmac_gguf.cpp matmul_q8.cpp

# Debug with verbose TMAC tracing:
g++ -std=c++17 -pthread -DTMAC_DEBUG -O0 -g -o tmac_gguf_debug tmac_gguf.cpp matmul_q8.cpp

g++ -std=c++17 -pthread -fsanitize=address -O1 -o tmac_gguf_asan tmac_gguf.cpp matmul_q8.cpp
```

### 6.2 Usage

```bash
# Single-token inference (prompt "Hi" = token 9707):
echo 9707 | ./tmac_gguf /tmp/model.tmac

# Autoregressive generation:
echo 9707 | ./tmac_gguf /tmp/model.tmac --generate 20

# FPGA simulation modes:
echo 9707 | ./tmac_gguf /tmp/model.tmac --fpga-int16  # INT16 sim (recommended)
echo 9707 | ./tmac_gguf /tmp/model.tmac --fpga-q8     # Q8_0 direct path sim

# Debug: dump layer intermediates:
echo 9707 | ./tmac_gguf /tmp/model.tmac --dump-layers

# Performance profiling:
echo 9707 | ./tmac_gguf /tmp/model.tmac --fpga-int16 --perf
```

### 6.3 Verification

The engine is verified against two independent ground truth sources:

1. **gguf Python library** (`scripts/ground_truth_v2.py`): Official dequantize → run FP32 forward pass
2. **llama.cpp patched main** (`scripts/llama_dump.c`): Dump logits after prompt eval

Results:
- Layer-by-layer hidden state: max diff < **0.002** across all 24 layers
- Final logits: max diff < **0.0003**
- Top-5 tokens: **5/5 match** with ground truth

### 6.4 KV Cache

- K cache: `[24 × 256 × 128]` (layers × seq_len × K_dim) — ~3.1 MB
- V cache: `[24 × 256 × 128]` — ~3.1 MB
- Maximum sequence length: 256 tokens
- GQA: 14 Q heads, 2 KV heads (7 queries per KV head)

### 6.5 Profiling (`--perf` flag)

Produces Chrome trace JSON (`/tmp/pipeline_trace.json`) and terminal breakdown showing
time spent per operation (attn_norm, attn_q, ffn_gate, etc.). Use for bottleneck analysis.

---

## 7. FPGA Accelerator (Verilog RTL)

### 7.1 Verilog Q8_0 Core (`verilog/matmul_q8_core.v`)

Custom Verilog RTL with 3-stage pipeline:

| Stage | Description |
|-------|-------------|
| 0 | BRAM address set {g,k}, act_reg[k] + smem[...] read |
| 1 | BRAM data arrives, dequant + multiply by activation → partial |
| 2 | Partial products accumulated into acc[64 × 48-bit] |

**Key design choices:**
- 8 parallel dequant+MAC lanes (one per row in bank group)
- BRAM-based weight storage (512×64-bit → 1 BRAM36)
- Single-bank accumulators (read-before-write, no copy block)
- Dequant: `q8_val * scale >> 8` (INT8 × UQ8.8 → INT16, 0 DSP)

**Performance:** 515 cycles/tile, ~413 ms/token @ 150 MHz

### 7.2 Top-level Integration (`verilog/matmul_q8_top.v`)

AXI4-Lite control interface + internal BRAM data buffers.

### 7.3 Legacy HLS Kernels (Deprecated)

| File | Type | Precision | Status |
|------|------|-----------|--------|
| `hls/matmul_q8.cpp` | Q8_0 direct path | INT16 systolic | **Deprecated** — superseded by Verilog |
| `hls/matmul_int16.cpp` | Pure INT16 | INT16×INT16→INT64 | **Deprecated** |
| `hls/matmul_int8.cpp` | Pure INT8 | INT8×INT8→INT32 | **Deprecated** (SwiGLU divergence) |

### 7.4 Resource Budget (Verilog, estimated)

| Resource | Available | Used (est.) | % |
|----------|-----------|-------------|---|
| DSP | 80 | 8 | 10% |
| LUT | 17,600 | ~2.5-3.5K | 14-20% |
| BRAM | 135 KB | 1 BRAM36 (36 KB) | ~27% |
| FF | 35,200 | ~6.5-9.7K | 37-55% |

*Post-synthesis numbers needed for exact usage.*

### 7.5 Estimated Performance

| Mode | Tiles | Cycles/Tile | Total Cycles | Time @ 150 MHz |
|------|-------|-------------|-------------|----------------|
| Decode (vecmul) | 120,596 | 515 | 62.1M | 413 ms |
| Prefill (batch) | 120,596 | 515 | 62.1M | 413 ms |

The optimistic case assumes ARM pre-dequants all weights once (cost amortized across tokens).

---

## 8. ARM Firmware (`firmware/`)

**Status: Aspirational — not yet cross-compiled or deployed.**

The firmware directory contains stub implementations that need to be rewritten for the
`matmul_q8` HLS IP. The current files (`tmac_fpga.hpp/.cpp`, `tmac_runtime.hpp/.cpp`,
`tmac_app.cpp`) have been updated to reflect the matmul_q8 interface but have not been
tested on real hardware.

Target memory map (512 MB DDR):
| Range | Size | Usage |
|-------|------|-------|
| 0x00000000 - 0x00FFFFFF | 16 MB | ARM code + stack |
| 0x01000000 - 0x01FFFFFF | 16 MB | Activation buffers |
| 0x02000000 - 0x07FFFFFF | 96 MB | KV Cache |
| 0x08000000 - 0x13FFFFFF | 192 MB | Model weights |
| 0x14000000 - 0x1FFFFFFF | 192 MB | Reserved |

---

---

## 10. Key Technical Details

### 10.1 GGUF Quantization Formats Implemented

- **Q5_0** (block: 32 elts, 22 B): `[d:f16][qh:u32][qs:u8×16]` — 5-bit per weight
- **Q6_K** (block: 256 elts, 210 B): Two 128-elt halves, each with ql/qh/scales — 6.5 bpw
- **Q8_0** (block: 32 elts, 34 B): `[d:f16][q:i8×32]` — 8.5 bpw
- **Q4_K** (block: 256 elts, 144 B): Super-block with sub-block scales — 4.5 bpw
- **F32**: Direct float
- **F16**: IEEE half-precision

### 10.2 Bugs Fixed (Historical)

See `docs/architecture.md` §6 for complete list. Notable:
- RMS norm: epsilon inside sqrt, not outside
- Q5_0: high-bit extraction off by 4 bits for elements 16-31
- Column-major indexing: `idx = col + row * input_dim` (not `row * cols + col`)
- Q6_K/Q4_K: complete rewrite to match ggml block layout

### 10.3 AXI Register Map (matmul_q8)

| Offset | Field | Access | Description |
|--------|-------|--------|-------------|
| 0x00 | AP_CTRL | R/W | [0]=ap_start, [1]=ap_done, [2]=ap_idle, [3]=ap_ready |
| 0x04 | GIE | R/W | [0]=Global Interrupt Enable |
| 0x08 | IER | R/W | [0]=IP Interrupt Enable |
| 0x0C | ISR | R/W | [0]=ap_done intr status (W1C) |
| 0x10 | CTRL_USER | R/W | [3]=op_vecmul, [5]=q8_path |
| 0x14 | STATUS | R | 0=IDLE, 1=RUNNING, 2=DONE, 3=ERROR |
| 0x18+ | A_ADDR, B_ADDR, C_ADDR | R/W | Auto-generated by Vitis HLS for m_axi |

---

## 11. Optimization Opportunities

### High Priority

1. **HLS Synthesis & Resource Feedback** — Run `make hls-q8` to get actual resource usage. The current estimates (~14K LUT, 64 DSP) need verification. Use `scripts/feedback_parser.py` to parse reports.

2. **ARM Firmware Implementation** — Rewrite `firmware/` to use the `matmul_q8` IP's actual register map. Current stubs (`tmac_fpga.hpp`) have correct interface signatures but no real AXI driver code.

3. **Vivado Block Design** — After HLS export, generate `hls_ip.tcl` and integrate into `vivado/block_design.tcl`. Connect AXI master ports to DDR via Zynq PS HP port.

### Medium Priority

4. **Precomputed Row Scales** — The `q8_logits_matmul_with_tensor` path (`matmul_q8.cpp`) can accept precomputed `row_max_abs` to avoid per-call full-tensor scanning. Generate once at load time, store as metadata tensor.

5. **Hybrid Prefill/Decode** — GQA attention uses `forward_layer_with_cache` for both prefill (batch) and decode (single token). The attention softmax path recomputes over all past positions each time — could cache the full log-sum-exp.

6. **Pipeline the ARM ARM↔FPGA Data Path** — Double-buffer tiles so ARM prepares the next tile while FPGA computes the current one. The `PipelineStats` struct in `fpga_sim.hpp` already models this.

### Low Priority

7. **Memory-Mapped GGUF Loading** — Skip TMAC conversion entirely. Load GGUF directly via `mmap` on Linux/Zynq. Saves the conversion step and one copy (~374 MB).

8. **INT4 K-quant Support** — If more model formats are needed, implement GGUF Q4_K dequant in C++ (currently only in Python). Block size 256, sub-block scales, dmin offset.

---

## 12. Handover Notes for Next AI Agent

### 12.1 What Works

- The C++ inference pipeline runs correctly on macOS (arm64) with TMAC format weights.
- All 24 layers and logits verified against Python ground truth.
- Three FPGA simulation modes: INT8, INT16, Q8_0 direct path.
- AXI-Lite register simulation with background FSM thread + interrupt.

### 12.2 What Needs Work

- **HLS synthesis** is the next step. Requires Xilinx Vitis HLS.
- After synthesis, check actual resource usage (DSP, LUT, BRAM) against estimates.
- If LUT > 85%, reduce unroll factor in `vecmul_1x64_q8` or `matmul_64x64`.
- If DSP > 80%, reduce systolic array size from 8×8 to 4×4 (cost: 2× latency).
- Vivado integration requires `hls_ip.tcl` from exported HLS IP.

### 12.3 Agent Workflow

The `docs/AGENTS.md` file contains the FPGA design workflow for AI agents:
- HLS/Vivado commands
- Three-layer feedback system (console, reports, Tcl queries)
- Optimization strategies for resource overuse
- Iteration loop (`make iterate`)

### 12.4 Key Files to Read First

| File | Why |
|------|-----|
| `sim/tmac_gguf.cpp` | Main inference engine — understand the full pipeline |
| `sim/fpga_sim.hpp` | FPGA accelerator simulation — MatmulAccel, AXI, timing |
| `hls/matmul_q8.cpp` | The HLS kernel that becomes FPGA bitstream |
| `docs/architecture.md` | Full architecture, quantization format details |
| `docs/AGENTS.md` | FPGA design iteration workflow for AI agents |

### 12.5 Environment

- **Host**: macOS (arm64)
- **C++ compiler**: Apple Clang (Xcode), also g++ via Homebrew
- **Python**: 3.x with `gguf`, `numpy`, `tokenizers` packages
- **FPGA tools**: Xilinx Vivado 2023.1 + Vitis HLS 2023.1
- **ARM target**: Cortex-A9, cross-compiled with `arm-linux-gnueabihf-g++`

---

## 13. References

- **Model**: [Qwen2-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct)
- **Quantization**: [llama.cpp GGUF](https://github.com/ggerganov/llama.cpp) — Q5_0, Q6_K, Q8_0, Q4_K formats
- **TMAC concept**: [T-MAC: CPU Renaissance via Table Lookup](https://github.com/microsoft/T-MAC) (Microsoft)
- **FPGA**: Xilinx Zynq-7000 TRM (UG585), Zynq 7010 datasheet
