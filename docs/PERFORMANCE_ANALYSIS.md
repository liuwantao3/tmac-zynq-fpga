# T-MAC FPGA Inference - Performance Analysis

## Project Overview

**Target Platform:** Xilinx Zynq 7010 (Z-turn Board)
- 512 MB DDR3
- 80 DSP slices
- 60× BRAM18K
- 150 MHz AXI interface

**Model:** Qwen2-0.5B-Instruct
- 24 layers
- Hidden dimension: 896
- Intermediate dimension: 4864
- Vocabulary: 151,936 tokens
- Context length: 256

**Quantization:** Q8_0 (8-bit)
- Original FP16: ~1 GB
- Q8_0: ~374 MB (fits in 512 MB DDR with KV cache)

**FPGA Resource Constraint:** 80 DSP slices (not 220 — Zynq 7010 vs 7020)
- 64 DSPs consumed by 8×8 systolic array (80% of budget)
- Q8_0 scale multiplication: LUT-based fixed-point (zero DSPs)
- See `docs/AGENTS.md` for resource usage details

---

## Architecture

### TMAC Format

Custom format for FPGA-efficient model storage:

```
struct TMACHeader {
    char magic[4] = "TMAC";
    uint64_t version;
    uint64_t n_tensors;
};

struct TensorHeader {
    uint64_t name_len;
    char name[name_len];           // Variable-length name
    uint64_t rows, cols;           // Dimensions
    uint32_t type;                // GGUF type enum
    uint64_t n_bytes;             // Data size
    uint8_t data[n_bytes];        // Raw tensor data
};
```

### Q8_0 Layout

Each Q8_0 block: 34 bytes = 2 bytes (FP16 scale) + 32 × INT8 values

```
┌──────────┬──────────┬──────────┐
│ scale(F16)│  int8[0] │  int8[1] │ ...
└──────────┴──────────┴──────────┘
   2 bytes    32 bytes total per block
```

### Q8_0 Dequantization (Fixed-Point LUT Path)

FPGA performs Q8→INT16 dequantization via LUT-based fixed-point multiply (zero DSPs):

```cpp
// ARM precomputes combined = block_scale / row_scale as UQ8.8 fixed-point
// FPGA receives combined + raw Q8 INT8 values
int24_t product = (int24_t)q8_val * (int24_t)combined;  // LUT multiply
int16_t w_int16 = product >> 8;                          // UQ8.8 → INT16
// Clamp to INT16 range if overflow
w_int16 = clamp(w_int16, -32768, 32767);

// INT16 systolic MAC (64 DSPs)
acc[i] += (int64_t)activation[k] * (int64_t)w_int16;
```

---

## Performance Profiling

### DSP Budget (Zynq 7010)

| Component | DSPs | % of 80 |
|-----------|------|---------|
| 8×8 systolic array | 64 | 80% |
| Control/logic | 0 | 0% |
| Q8_0 scale multipliers | 0 (LUT) | 0% |
| **Total** | **64** | **80%** |

All Q8_0→INT16 dequantization scale multiplications use **LUT-based UQ8.8 fixed-point** arithmetic via `config_bind -mul_style luts`. Zero DSPs consumed for scale operations.

### Profiler Implementation

Located in `tmac_gguf.cpp` (lines 28-133):

**Chrome Trace Format:**
- `PROFILE_SCOPE(name)` macro wraps code blocks
- Uses `std::chrono::high_resolution_clock` for wall-clock timing
- Outputs to `/tmp/pipeline_trace.json` for Chrome trace viewer

**Timing Aggregation:**
- Stack-based begin/end event pairing
- Sorts by total time descending
- Reports: Time (ms), Calls, Avg (ms), Share (%)

### Key Profiler Macros

```cpp
#define PROFILE_SCOPE(name) TraceScope CONCAT(_trace_, __LINE__)(name)
```

### Running with Profiler

```bash
./tmac_gguf <model.tmac> --fpga-q8 --perf --generate N
```

Output:
```
[PERF — PIPELINE TIMING BREAKDOWN]
  Operation               Time (ms)    Calls   Avg (ms)    Share
  -----------------------------------------------------------------
  forward_all_layers     18392.12        8 2299.0151   43.0%
  ffn_down                8149.03      216  37.7270   19.0%
  ...

[TRACE] Wrote Chrome trace: /tmp/pipeline_trace.json
```

### Important Caveats

1. **Nested Scopes**: `forward_all_layers` wraps all 24 layer calls in ONE scope. Sub-operations (ffn_down, attn_q, etc.) are profiled INSIDE `forward_layer_with_cache`. Wall-clock times will overlap.

2. **Wall-Clock vs CPU-Time**: Profiler uses `std::chrono::high_resolution_clock::now()` for wall-clock time, not CPU time.

---

## Current Status

### What's Working

| Component | Status | Notes |
|-----------|--------|-------|
| TMAC loader | ✅ | Fixed name_len truncation bug |
| Q8_0 path | ✅ | matmul_fpga_q8 for all matmuls |
| row_max_abs | ✅ | Precomputed for token_embd.weight only |
| Process prompt | ✅ | 1 token × 24 layers |
| Token generation | ✅ | 8 tokens × 24 layers |
| Benchmarking | ✅ | --perf flag for profiling |

### Bug Fixes

**SIGSEGV Crash (FIXED)**
- **Root cause**: When tensor name ≥ 128 bytes, code truncated to 127 but never consumed remaining bytes from file stream
- **Fix**: Read 127 bytes, null-terminate, then `fseek(f, name_len - 127, SEEK_CUR)` to skip remainder
- **Location**: `tmac_gguf.cpp:1053-1068`

**Pipelined Version Removed (BROKEN)**
- **Issue**: Ring buffer approach with FPGA simulation sleep caused hangs on large matmuls
- **Root cause**: `Q8_TILE_CYCLES * 6.67µs` sleep per tile caused CPU to block when ring buffer filled
- **Fix**: Removed ~250 lines of broken pipelined code, using `matmul_fpga_q8` instead

---

## Benchmark Results

### Baseline: Q8 with row_max_abs (8 tokens)

```
[FPGA TIMING SUMMARY]
  Tiles: 1052128   MACs: 4309516288  CPU: 22105.20 ms  NAIVE: 0.19 Gop/s
  FPGA: 120399552 cycles @ 150 MHz = 802663.68 us  Speedup: 27.5x

[PERF — PIPELINE TIMING BREAKDOWN]
  Operation               Time (ms)    Calls   Avg (ms)    Share
  -----------------------------------------------------------------
  forward_all_layers     18465.39        8 2308.1734   43.1%
  ffn_down                8017.51      216  37.1181   18.7%
  ffn_gate                5381.76      216  24.9156   12.5%
  ffn_up                  5248.31      216  24.2977   12.2%
  process_prompt          2299.99        1 2299.9900    5.4%
  logits                  1356.06        8 169.5071    3.2%
  attn_q                   925.92      216   4.2866    2.2%
  attn_output              920.93      216   4.2636    2.1%
  attn_k                   131.05      216   0.6067    0.3%
  attn_v                   128.77      216   0.5962    0.3%
```

### Without row_max_abs (8 tokens, for comparison)

```
  logits                  3703.51        8 462.9391    8.1%
  Total                  45497.94 ms
```

---

## Optimization Gains

### Measured Improvements

| Optimization | Before | After | Improvement |
|-------------|--------|-------|-------------|
| **row_max_abs for logits** | 3703 ms | 1356 ms | **63.4% reduction** |
| **Total (8 tokens)** | 45498 ms | 42885 ms | **5.7% reduction** |

### Where Time Goes

| Component | Time (ms) | Share | row_max_abs? |
|-----------|-----------|-------|-------------|
| FFN (down+up+gate) | ~18648 | 43.4% | ❌ No |
| Attention matmuls | ~2188 | 5.1% | ❌ No |
| Logits matmul | ~1356 | 3.2% | ✅ Yes |
| Other (RoPE, softmax, etc.) | ~100 | 0.2% | N/A |

### Theoretical Full row_max_abs Coverage

If row_max_abs were precomputed for ALL matmuls (FFN + attention):

| Component | Current | With row_max_abs | Savings |
|-----------|---------|------------------|---------|
| FFN (43.4%) | ~18648 ms | ~6900 ms | ~11748 ms |
| Attention (5.1%) | ~2188 ms | ~810 ms | ~1378 ms |
| Logits (3.2%) | ~1356 ms | ~500 ms | ~856 ms |
| **Total** | ~42885 ms | ~22700 ms | **~20185 ms (47%)** |

---

## Inference Pipeline

### Two Phases

**1. Prompt Processing (`process_prompt`)**
- Processes all input tokens once
- Computes and caches K/V values
- For 1 token × 24 layers = 24 forwards

**2. Token Generation (`generate`)**
- Generates new tokens one at a time
- Reuses cached K/V (attention is cheaper)
- For 8 tokens × 24 layers = 192 forwards

```cpp
void generate(float* hidden, float* logits, int prompt_len, int n_tokens, int top_k) {
    for (int gen = 0; gen < n_tokens; gen++) {
        int pos = prompt_len + gen;
        get_logits_q8(logits, hidden);           // Compute logits
        int next_token = sample_token(logits);   // Sample
        process_embedding(hidden, next_token);    // Add to sequence
        forward_all_layers(hidden, pos);           // Forward through all layers
    }
}
```

---

## Model Tensors

### Qwen2-0.5B Structure (303 tensors)

| Tensor | Shape | Type | Size |
|--------|-------|------|------|
| token_embd.weight | 151936 × 896 | Q8_0 | 138 MB |
| token_embd.weight_row_max_abs | 151936 × 1 | F32 | 0.6 MB |
| blk.{n}.attn_q.weight | 896 × 896 | Q8_0 | 2.7 MB |
| blk.{n}.attn_k.weight | 128 × 896 | Q8_0 | 0.4 MB |
| blk.{n}.attn_v.weight | 128 × 896 | Q8_0 | 0.4 MB |
| blk.{n}.attn_output.weight | 896 × 896 | Q8_0 | 2.7 MB |
| blk.{n}.ffn_gate.weight | 4864 × 896 | Q8_0 | 2.9 MB |
| blk.{n}.ffn_up.weight | 4864 × 896 | Q8_0 | 2.9 MB |
| blk.{n}.ffn_down.weight | 896 × 4864 | Q8_0 | 3.4 MB |
| output_norm.weight | 896 × 1 | F32 | 3.5 KB |
| blk.{n}.attn_norm.weight | 896 × 1 | F32 | 3.5 KB |
| blk.{n}.ffn_norm.weight | 896 × 1 | F32 | 3.5 KB |

**Total:** 374.3 MB (including 13 row_max_abs tensors)

---

## Future Optimization Suggestions

### Priority 1: Precompute row_max_abs for All Layers

**Impact:** ~47% total time reduction (~20 seconds saved per 8 tokens)

**Effort:** Low - add 13 more tensor precomputations during model conversion

**Implementation:**
1. During TMAC generation, compute row_max_abs for all Q8 weight matrices
2. Store as separate tensors named `{tensor}_row_max_abs`
3. `matmul_fpga_q8` already supports `row_max_abs` parameter

### Priority 2: FFN Matmuls on Dedicated FPGA Path

**Impact:** Currently 43.4% of time

**Current:** FFN uses `matmul_fpga` which dequantizes INT8→FP32→INT16

**Potential:** Custom Q8→INT16 kernel optimized for FFN shapes (896×4864)

### Priority 3: Attention on FPGA

**Impact:** ~5% of time

**Current:** Attention Q/K/V projections use `matmul_fpga_int16` (FP16 path)

**Potential:** Dedicated Q8 kernels for attention matmuls

### Priority 4: Softmax on FPGA

**Impact:** Vocab softmax is expensive (151,936 elements)

**Current:** `sample_token` does partial_sort + exp + normalize on CPU

**Potential:** Parallel softmax kernel for top-k selection

### Priority 5: Q6_K Support

**Impact:** Better accuracy/compression ratio

**Current:** Q8_0 at 8-bit

**Potential:** Q6_K at ~6.5-bit effective (better accuracy, similar compute)

---

## File Structure

```
/Users/arctic/fpga/
├── sim/
│   ├── tmac_gguf.cpp       # Main inference engine, profiler, TMAC loader
│   ├── matmul_q8.cpp        # Q8 matmul implementations
│   ├── fpga_sim.hpp         # FPGA simulation utilities, TileCycleBudget
│   └── model.tmac           # Qwen2-0.5B Q8_0 model (374.3 MB)
├── hls/                     # Vivado HLS for FPGA bitstream
├── firmware/                 # FPGA firmware
├── docs/                     # Documentation
└── scripts/                  # Build/iteration scripts
```

---

## Commands Reference

```bash
# Build
cd /Users/arctic/fpga/sim
clang++ -O2 -std=c++17 -o tmac_gguf tmac_gguf.cpp matmul_q8.cpp -lm -lpthread

# Run with profiling
echo "0" | ./tmac_gguf /Users/arctic/fpga/models/model.tmac --fpga-q8 --perf --generate 8

# Run without profiling
echo "0" | ./tmac_gguf /Users/arctic/fpga/models/model.tmac --fpga-q8 --generate 8

# Toggle row_max_abs (edit matmul_q8.cpp line ~460)
const bool USE_PRECOMPUTED = true;  // or false
```

---

## Glossary

| Term | Definition |
|------|------------|
| **TMAC** | Custom format: magic + header + tensors for FPGA |
| **Q8_0** | 8-bit quantization with FP16 scale per 32-element block |
| **row_max_abs** | Precomputed max absolute value per row (for 1× data transfer) |
| **KV Cache** | Cached key/value activations for attention efficiency |
| **RoPE** | Rotary Position Embedding - positional encoding method |
| **FFN** | Feed-Forward Network - MLP layers in transformer |
| **GEMV** | General Matrix-Vector Multiply |
| **GEMM** | General Matrix-Matrix Multiply |
