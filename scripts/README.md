# Scripts Directory

Utilities for model conversion, ground truth generation, verification, and FPGA design iteration.

## Model Conversion

| Script | Purpose |
|--------|---------|
| `extract_tmac.py` | Converts GGUF model files to TMAC binary format for C++ inference (`tmac_gguf.cpp`). Creates `.tmac` files with all tensors (Q4_K, Q8_0, FP32, etc.) |

## Ground Truth & Verification

| Script | Purpose |
|--------|---------|
| `ground_truth_v2.py` | Complete Qwen2-0.5B FP32 inference via gguf Python library. Generates layer-by-layer hidden states for cross-validation |
| `py_tmac_vec.py` | Vectorized Python TMAC reference inference. Reads TMAC format, matches C++ exactly (same dequant, same arithmetic) |
| `verify_layers_fast.py` | Compares C++ `--dump-layers` output against Python ground truth layer by layer. Reports max diff per layer |

## FPGA Design Iteration

| Script | Purpose |
|--------|---------|
| `design_iteration.sh` | Orchestrates HLS synthesis and/or Vivado implementation. Run via `bash scripts/design_iteration.sh hls` / `vivado` / `all` |
| `feedback_parser.py` | Parses HLS/Vivado reports into structured JSON with metrics and optimization recommendations. Run via `python3 scripts/feedback_parser.py` |

## Integration Test

| Script | Purpose |
|--------|---------|
| `test_integration.sh` | Full system test: C++ AXI-Lite buffer API + Q4K Verilog core + Q8_0 Verilog core + (optionally) full model inference |

## Usage

```bash
# Model conversion
python3 scripts/extract_tmac.py /path/to/model.gguf /tmp/model.tmac

# Ground truth generation
python3 scripts/ground_truth_v2.py /tmp/model.tmac 9707

# Layer verification (after running C++ with --dump-layers)
python3 scripts/verify_layers_fast.py /path/to/ground_truth /tmp/cpp_layers

# FPGA design iteration
bash scripts/design_iteration.sh hls      # HLS synthesis
bash scripts/design_iteration.sh vivado   # Vivado implementation
bash scripts/design_iteration.sh all      # Full iteration loop with feedback

# Integration test
bash scripts/test_integration.sh
```

## Removed Scripts

The following scripts were removed as superseded or incomplete:
- `gguf_inference.py` — superseded by `ground_truth_v2.py`
- `gguf_layer_inference.py` — incomplete forward pass (only embedding + norm)
- `compare_weights.py` — dequant debugging, no longer needed
- `llama_dump.c` — depended on llama.cpp, not part of current workflow
