# FPGA T-MAC Project Progress Summary

> **Note**: This document describes the historical INT4 approach (abandoned). The current implementation uses GGUF-sourced Q5_0/Q8_0 weights with the Q8_0 direct-path HLS kernel (`matmul_q8`). See `README.md` and `docs/architecture.md` for the current design.

## Executive Summary

This document summarizes the development of a T-MAC (Table-based Multiplication-Accumulation) inference engine for Qwen2.5 0.5B model on Zynq 7010 FPGA. The project has evolved from debugging C++ inference to exploring a new approach: leveraging llama.cpp's proven GGUF quantization with the FPGA accelerator.

---

## 1. Project Context

### Hardware Constraints (Zynq 7010)
- ARM Cortex-A9 processor
- 80 DSPs available
- 240KB BRAM
- 28K LUTs
- 512MB DDR

### Model Configuration
- Qwen2.5 0.5B (24 layers, 896 hidden dim, 14 heads, head_dim=64)
- Target: INT4 quantization with N=64 blocks (later N=4)

### Original Goal
Build an end-to-end C++ inference simulation that matches Python reference, then deploy to FPGA with custom matrix multiplication accelerator.

---

## 2. Development Progress

### 2.1 Files Created/Modified

#### C++ Simulation Files
| File | Purpose |
|------|---------|
| `sim/tmac_e2e_v13.cpp` | Main inference engine (BLOCK_SIZE=4) |
| `sim/tmac_e2e_v13` | Compiled binary |
| `sim/tmac_debug_v13.cpp` | Debug version |

#### Python Reference
| File | Purpose |
|------|---------|
| `scripts/python_inference_v13.py` | Full Python INT4 inference |
| `scripts/verify_v13.py` | Weight verification |
| `scripts/generate_ground_truth_hi.py` | Ground truth generator |

#### Ground Truth
| Directory | Contents |
|-----------|-----------|
| `sim/ground_truth_hi/` | 24-layer intermediate values for "Hi" prompt |

### 2.2 Key Fixes Applied

1. **Tokenizer Fix**: Changed from ASCII to Qwen tokens [39, 72] for "Hi"
2. **Architecture Fix**: NUM_HEADS=14, HEAD_DIM=64, NUM_KV_HEADS=2
3. **SiLU Fix**: Changed from `1/(1+exp(-x))` to `x/(1+exp(-x))`
4. **Compiler Fix**: Added `-fno-strict-aliasing` for -O2 optimization
5. **Ground Truth Fix**: Regenerated with correct SiLU

### 2.3 Binary Files

| Binary | Size | BLOCK_SIZE | Status |
|--------|------|------------|--------|
| `qwen2.5-fpga-int4-v2.bin` | 311MB | 4 | Active - causes explosion |
| `qwen2.5-fpga-int4.bin` | 247MB | 64 | Removed - caused confusion |

---

## 3. Technical Findings

### 3.1 Current Problem: INT4 Value Explosion

**Observation**: When using BLOCK_SIZE=4 binary, values explode around layer 7-9:

| Layer | INT4 Sum | FP16 GT Sum | Ratio |
|-------|----------|-------------|-------|
| 0 | 3.13 | 3.52 | 0.11 |
| 6 | -21.96 | -22.05 | 0.00 |
| 7 | 12.91 | -32.19 | 1.40 |
| 8 | -32.46 | -11.23 | 1.89 |
| 9 | -83.05 | 16.30 | 6.10 |
| 10 | 39.65 | -13.47 | 3.94 |
| 13 | -1.5M | 2.5M | 1.59 |
| 14 | 31B | overflow | - |
| 17 | NaN | NaN | - |

**Root Cause**: INT4 quantization error accumulates multiplicatively through FFN residual connections. The 4-bit precision (16 discrete levels) is insufficient for deep networks.

### 3.2 v13 C++ vs Python Comparison

**Result**: C++ and Python match EXACTLY for all 24 layers (diff < 0.001)

This confirms the C++ implementation is correct for INT4 inference. The explosion is inherent to the quantization, not implementation bugs.

### 3.3 GGUF Analysis

**Ollama's Qwen2.5 0.5B** (analyzed):
- Format: GGUF version 3
- Size: 379 MB (~6 bpw) → suggests Q5_0 or Q5_1
- Block size: 32 elements (vs our 16)
- Quantization: proven stable in production

### 3.4 llama.cpp Quantization Approaches

| Type | Block Size | Bits | BPW | Notes |
|------|------------|------|-----|-------|
| Q4_0 | 32 | 4 | 4.34 | Fixed zero-point |
| Q4_1 | 32 | 4 | 4.78 | Variable zero-point |
| Q5_0 | 32 | 5 | 5.21 | Higher precision |
| Q5_1 | 32 | 5 | 5.50 | Best balance |
| Q4_K | 256 | 4 | ~4.0 | K-quantization |

---

## 4. Current Architecture

### 4.1 Inference Flow (Current)

```
Input: "Hi" tokens [39, 72]
         │
         ▼
  ┌─────────────┐
  │ Embedding   │  INT4 dequantize (BLOCK_SIZE=4)
  │ Layer       │
  └─────────────┘
         │
         ▼
  ┌────────────────────────────────────────┐
  │ Layer 0-23 (repeated)                   │
  │  - RMSNorm                              │
  │  - Q/K/V projection (INT4 matmul)      │
  │  - RoPE (no quantization)              │
  │  - Attention (FP32)                     │
  │  - RMSNorm                              │
  │  - O projection (INT4)                 │
  │  - FFN: Gate×Up×Silu (INT4)            │
  │  - Down projection (INT4)              │
  └────────────────────────────────────────┘
         │
         ▼
  ┌─────────────┐
  │ Logits      │  Embedding.T @ hidden
  │ (INT4)      │
  └─────────────┘
         │
         ▼
  Output: Next token
```

### 4.2 FPGA Acceleration Strategy

Current design uses N=64 blocks to fit within 80 DSP constraint. The FPGA accelerator performs INT16 matrix multiplication via Q8_0 direct path with LUT-based scale multipliers (64 DSPs for 8×8 systolic array).

---

## 5. Proposed New Approach

### 5.1 Goal
Port llama.cpp's GGUF model loading and quantization to utilize the FPGA accelerator without converting to proprietary format.

### 5.2 Rationale
1. GGUF is proven in production (Ollama, llama.cpp)
2. Q5 quantization is more stable than Q4 for small models
3. Larger block size (32) provides better precision
4. Avoids conversion overhead and format maintenance

### 5.3 Implementation Plan

#### Phase 1: Python Reference (Recommended)
1. Load GGUF model using llama.cpp's Python bindings or pyllama
2. Implement forward pass using FP16/BF16 weights from GGUF
3. Generate ground truth for "Hi" prompt
4. Verify against Ollama output

#### Phase 2: C++ Implementation
1. Parse GGUF format (metadata, tensor headers)
2. Implement dequantization for Q4_0/Q5_0
3. Hook into existing matrix multiplication
4. Match Python ground truth

#### Phase 3: FPGA Integration
1. Keep weights in GGUF format
2. Use FPGA accelerator for FP16 matmul (not INT4)
3. Implement per-block dequantization in ARM

### 5.4 Trade-offs

| Aspect | Current INT4 | New GGUF |
|--------|--------------|----------|
| Weight size | 311MB | 379MB |
| Precision | 4-bit | 5-8 bit |
| Stability | Explodes at layer 9 | Stable |
| FPGA DSP | 80 (80% used) | 64 (systolic) |
| Model conversion | Required | None |

---

## 6. Development Methodology (Per User Preference)

Following the successful debugging pattern from v13 development:

### Step 1: Ground Truth
- Generate with Python using FP16 GGUF weights
- Save intermediate values for all 24 layers
- Verify against Ollama CLI output

### Step 2: Python Inference
- Implement full forward pass
- Use dequantized weights (FP16 from GGUF)
- Match ground truth exactly

### Step 3: C++ Implementation
- Parse GGUF format
- Replicate Python logic
- Verify against Python ground truth

### Step 4: FPGA Acceleration
- Profile existing matmul
- Decide on INT4 vs FP16 on FPGA
- Optimize data movement

---

## 7. Files to Keep/Remove

### Keep
- `sim/tmac_e2e_v13.cpp` - Reference for correct inference logic
- `scripts/python_inference_v13.py` - Reference for INT4 path
- `scripts/generate_ground_truth_hi.py` - Ground truth generator
- `sim/ground_truth_hi/` - Ground truth data
- `models/qwen2.5-fpga-int4-v2.bin` - Current INT4 binary

### Remove
- All v3-related files (completed)

---

## 8. Historical Next Steps (Resolved)

These items were from the INT4 era. The current project status:

- ✅ **GGUF-based ground truth**: Generated using `scripts/ground_truth_v2.py` and `scripts/gguf_inference.py`
- ✅ **C++ inference engine**: `sim/tmac_gguf.cpp` — verified matching ground truth across all 24 layers
- ✅ **GGUF→TMAC converter**: `scripts/extract_tmac.py` working
- ✅ **HLS kernel**: `hls/matmul_q8.cpp` — Q8_0 direct path with LUT-based scale multipliers
- ❌ **FPGA deployment**: Not yet synthesized (needs Vitis HLS)

---

## 9. Open Questions (Resolved)

1. **INT16 on FPGA** (Q8_0 direct path: INT16 systolic, LUT dequant)
2. **Q5_0/Q6_K/Q8_0** mixed quantization from GGUF
3. **TMAC format** loads tensors into simulated DDR (`malloc` + memcpy)
4. **~214× speedup** vs CPU FP32 at 150 MHz (estimated)

---

*Last Updated: 2026-05-13*
*Project: Qwen2-0.5B on Zynq 7010 FPGA*