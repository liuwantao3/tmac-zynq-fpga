# Session Summary — May 8, 2026

## Goal
Debug C++ TMAC inference for Qwen2-0.5B-Instruct by comparing layer-by-layer outputs with gguf Python library / llama.cpp ground truth, then achieve correct end-to-end generation. Next: evaluate/improve the C++ inference and simulate the FPGA matrix multiply accelerator.

## Constraints & Preferences
- Standalone C++ without llama.cpp dependency
- TMAC model format (weights bit-identical to source GGUF)
- Must fit in 512MB DDR (373.7 MB used)
- llama.cpp and gguf Python library as ground truth references (agree to max diff 0.0018)
- No Vivado installed (no disk space) — FPGA simulation must work without real hardware

## Current State — Working

### C++ Inference Engine
- **File**: `sim/tmac_gguf.cpp` (748 lines, standalone C++17, no external deps)
- **Binary**: `sim/tmac_gguf` (arm64, Mac)
- **Usage**:
  - Single token: `echo 9707 | ./sim/tmac_gguf /tmp/model.tmac` → top-10 logits JSON
  - Generation: `echo "9707 3838" | ./sim/tmac_gguf /tmp/model.tmac --generate 20`
  - Layer dump: `echo 9707 | ./sim/tmac_gguf /tmp/model.tmac --dump-layers`
- **Capabilities**:
  - Reads tokens from stdin, writes logits to `/tmp/cpp_logits.bin`
  - Autoregressive generation (`--generate N`), prints token IDs to stdout
  - KV cache (MAX_SEQ_LEN=256)
  - All quantization: Q5_0, Q6_K, Q4_K, Q8_0, F32, F16
  - GQA attention (14 query heads, 2 KV heads)
  - RoPE, SiLU, RMS norm (eps=1e-6)
- **Verified**: matches gguf library ground truth across all 24 layers (max diff < 0.002) and logits (max diff < 0.0003)

### Files
```
/Users/arctic/fpga/
├── sim/
│   ├── tmac_gguf.cpp      # Main C++ inference engine
│   └── tmac_gguf           # Compiled arm64 binary
├── scripts/
│   ├── extract_tmac.py     # GGUF → TMAC converter
│   ├── ground_truth_v2.py  # Ground truth via gguf.quants.dequantize + tokenizer
│   ├── gguf_inference.py   # Ground truth via GGUFReader (manual dequant)
│   ├── gguf_layer_inference.py  # Vectorized ground truth
│   ├── py_tmac_vec.py      # Vectorized Python TMAC (mirrors C++)
│   ├── verify_layers_fast.py    # Layer-by-layer verification
│   ├── compare_weights.py  # Dequant comparison tool
│   ├── llama_dump.c        # Reference patch for llama.cpp
│   ├── feedback_parser.py  # FPGA HLS/Vivado feedback parser
│   └── design_iteration.sh # FPGA design iteration workflow
├── firmware/               # ARM runtime (aspirational FPGA)
│   ├── tmac_app.cpp
│   ├── tmac_fpga.cpp / .hpp
│   └── tmac_runtime.cpp / .hpp
├── hls/                    # HLS accelerator (aspirational)
│   ├── matmul_int8.cpp
│   ├── matmul_int8.hpp
│   ├── script.tcl
│   └── hls_config.tcl
├── vivado/
│   └── block_design.tcl
├── models/
│   └── qwen2-0_5b-instruct-q4_k_m.gguf  # Source GGUF (~392 MB)
├── docs/
│   ├── architecture.md     # Full knowledge document (312 lines)
│   ├── AGENTS.md           # FPGA dev workflow
│   └── PROGRESS_SUMMARY.md # Historical progress tracking
├── learning/               # User's personal learning materials
├── Makefile                # FPGA workflow targets
└── SESSION_SUMMARY.md      # This file
```

### Model Details (Qwen2-0.5B-Instruct)
| Parameter | Value |
|-----------|-------|
| Hidden dim | 896 |
| Intermediate dim (FFN) | 4864 |
| Vocab size | 151936 |
| Num layers | 24 |
| Num attention heads | 14 |
| Head dim | 64 |
| Num KV heads | 2 (GQA) |
| RMS norm eps | 1e-6 |
| RoPE theta | 1000000.0 |
| Tie word embeddings | true (lm_head = token_embd.weight) |

### Quantization Breakdown
| Type | Count | Tensors |
|------|-------|---------|
| Q5_0 | 132 | Most weight matrices |
| Q6_K | 12 | `blk.*.ffn_down.weight` |
| Q8_0 | 2 | `token_embd.weight`, `attn_v.weight` |
| Q4_K | rest | Remaining weight matrices |
| F32 | few | Norm/bias tensors |

### TMAC Model
- Source GGUF: `/Users/arctic/Downloads/qwen2-0_5b-instruct-q4_k_m.gguf`
- TMAC binary: `/tmp/model.tmac` (373.7 MB, 290 tensors)

### Ground Truth
- Token 9707 ("Hello") → top-1 token 11 (",") at logit 15.748
- Token 3838 ("What") → top-1 token 374 (" is") at logit 19.025
- llama.cpp patched at `/Users/arctic/llama.cpp/examples/main/main.cpp` line 660-668
- Ground truth logits at `/tmp/llama_logits.bin`
- C++ logits at `/tmp/cpp_logits.bin`

### Key Dequantization Formulas
All dequant functions in `sim/tmac_gguf.cpp` match ggml reference:
- `ggml/src/ggml-quants.c` (llama.cpp)
- `gguf/quants.py` (Python gguf library)

### Bugs Found & Fixed (6 total)
1. **RMS norm formula**: `x * w / sqrt(mean + eps)` — was dividing after sqrt, before adding eps
2. **RMS norm epsilon**: 1e-5 → 1e-6 (model requires 1e-6)
3. **Q5_0 dequant**: qh shift for elements 16-31 was off by 4 bits
4. **Q6_K dequant**: wrong block layout (256 elements, not 128 halves; wrong ql byte indexing for groups 1/3)
5. **Q4_K dequant**: missing dmin and sub-block scales in both C++ and Python; wrong byte offsets
6. **read_tensor column-major indexing**: GGUF stores 2D as flat[input + output * input_dim]; old code assumed row-major

---

## Next Goal: FPGA Matrix Multiply Simulation

### Objective
Replace the naive C++ matmul loops in `sim/tmac_gguf.cpp` with calls to a simulated version of the FPGA systolic array accelerator (`hls/matmul_int8.cpp`), running the HLS kernel logic in software to verify correctness.

### Why
- Can't test on real Zynq 7010 (no disk space for Vivado)
- Want to validate the accelerator design works before investing in synthesis
- Eventually: the accelerator accelerates the GEMM-heavy parts (QKV projections, output projection, FFN gate/up/down)

### FPGA Accelerator Design (from hls/matmul_int8.cpp)
- **Architecture**: 8×8 systolic array (64 DSPs)
- **Data types**: INT8 input, INT8/INT4 weights, INT32 accumulator
- **Sub-block size**: 8×8, tiles up to N=64
- **Modes**: Matrix-Matrix (MatMul) and Matrix-Vector (VecMul), INT4 or INT8
- **Interface**: AXI control/status registers, AXI master for DDR data transfer

### Simulation Approach
1. Write a software simulation of the HLS kernel: same bit-accurate INT8×INT8→INT32 computation, same blocking structure (8×8 systolic)
2. In `tmac_gguf.cpp`, replace matmul calls with: dequant weights → quant activations to INT8 → call simulated accelerator → dequant result back to FP32
3. Verify output still matches ground truth (or measure quantization error introduced by INT8 path)

### Key Design Decisions Needed
- How to handle mixed precision: weights are Q5_0/Q6_K/Q4_K/Q8_0, activations are FP32. The accelerator does INT8×INT8. Need quantization path: FP32 activation → INT8, and FP32 → INT8 weight (from dequantized weight).
- Or: keep weights dequantized to FP32, quantize both to INT8 at matmul time.
- Block size matching: the HLS kernel tiles at 8×8. Need to handle arbitrary dimensions (896, 4864) by padding or iterating over tiles.
- VecMul vs MatMul modes: decode tokens are vector×matrix, prefill is matrix×matrix.

### Expected Work
1. Read and understand `hls/matmul_int8.cpp` kernel design
2. Write `sim/fpga_sim.cpp` (or add to tmac_gguf.cpp) — bit-accurate software model
3. Create integration test: run a single matmul through both naive and FPGA-sim paths, compare
4. Incrementally swap matmul calls in tmac_gguf.cpp to use FPGA sim
5. Verify end-to-end output still matches ground truth
6. Measure/characterize any numerical differences from INT8 quantization
