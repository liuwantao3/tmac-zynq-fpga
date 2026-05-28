# Descriptor Orchestrator

Toolchain for building FPGA descriptor chains for LLM inference.

## Overview

This toolchain allows declarative configuration of the full inference pipeline, including:
- Token embedding
- Transformer layers (with mixed quantization)
- Logits computation

It compiles JSON configurations into binary descriptor chains that the FPGA can execute.

## Why This Toolchain?

Traditional approach: Hard-coded C++ dispatch based on tensor types.

This approach: Declarative JSON configuration that's:
- **Model-agnostic**: Works with any GGUF model (Qwen, Llama, Mistral, etc.)
- **FPGA-capability-aware**: Easy to move ops from CPU→FPGA or vice versa
- **Version-controlled**: Entire pipeline in JSON files

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  JSON Configuration                                    │
│  ┌──────────────────────────────────────────────────┐ │
│  │ model.json                                        │ │
│  │ ├── embedding stage                              │ │
│  │ ├── 28 layer references                        │ │
│  │ └── logits stage                               │ │
│  └──────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│  Pipeline Compiler (compile_pipeline.py)              │
│                                                  │
│  Input:  model.json                                │
│  Output: pipeline.bin (binary descriptors)         │
└──────────────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│  FPGA Execution                                      │
│                                                  │
│  Descriptors drive the FPGA state machine             │
│  CPU handles CPU_OP (type=15) interruptions        │
└──────────────────────────────────────────────────────────┘
```

## Files

| File | Purpose |
|------|---------|
| `compile_pipeline.py` | Main compiler (single layer or full pipeline) |
| `build_chain.py` | Multi-layer chain builder (legacy) |
| `layer_config.schema.json` | JSON Schema for layer configs |
| `examples/model.json` | Full Qwen2-0.5B pipeline config |
| `examples/stages/` | Embedding and logits stage definitions |
| `examples/layers/` | 28 layer configurations |

## Usage

### Compile Full Pipeline

```bash
# Dump summary (show all ops without generating binary)
python3 compile_pipeline.py examples/model.json --dump

# Compile to binary
python3 compile_pipeline.py examples/model.json -o /tmp/pipeline.bin
```

### Compile Single Layer

```bash
# From JSON config
python3 compiler.py examples/layers/layer_00.json -o out.bin

# From YAML config
python3 compiler.py examples/layer.yaml -o out.bin
```

### Build Multi-Layer Chain (all 28 layers)

```bash
python3 build_chain.py -o /tmp/full_chain.bin
```

### Build Individual Layers

```bash
# Generate all 28 layer configs
for i in $(seq 0 27); do
    python3 build_chain.py -n 1 -o layer_$i.json
done
```

## Configuration Schema

### model.json

Top-level pipeline configuration:

```json
{
  "model": "qwen2-0.5b",          // Model name
  "num_layers": 28,               // Number of transformer layers
  "hidden_dim": 896,             // Hidden dimension (H)
  "intermediate_dim": 4864,      // FFN intermediate dimension
  "vocab_size": 151936,          // Vocabulary size
  "num_heads": 14,              // Query heads
  "num_kv_heads": 2,            // KV heads
  "head_dim": 64,               // Head dimension
  "max_seq_len": 32768,         // Maximum sequence length
  
  "stages": [
    {
      "name": "embedding",
      "ops": [...]
    },
    {
      "name": "layer_00",
      "ref": "layers/layer_00.json"
    },
    // ... more layers ...
    {
      "name": "logits",
      "ops": [...]
    }
  ]
}
```

### Stage Definition

A stage contains ordered operations:

```json
{
  "name": "embedding",
  "ops": [
    {
      "name": "token_embedding",
      "weight": "token_embd.weight",
      "in": "input_ids",
      "out": "hidden",
      "quant_type": "Q8_0"
    }
  ]
}
```

### Layer Reference

Each layer is a separate file:

```json
{
  "layer": 0,
  "ffn_down_type": "Q6_K",    // Even layers: Q6_K, Odd: Q4_K
  "ops": [
    {"name": "rms_norm", ...},
    {"name": "matmul_q", ...},
    {"name": "matmul_k", ...},
    {"name": "matmul_v", ...},
    {"name": "softmax_attention", ...},
    {"name": "matmul_output", ...},
    {"name": "residual_add", ...},
    {"name": "matmul_gate", ...},
    {"name": "matmul_up", ...},
    {"name": "swiglu", ...},
    {"name": "matmul_down", "quant_type": "Q6_K"},
    {"name": "residual_add", ...}
  ]
}
```

## Operation Names

### CPU Operations (type=15)

These trigger CPU interrupts:

| Operation | Description |
|----------|-------------|
| `rms_norm` | RMSNorm normalization |
| `softmax_attention` | Attention softmax + matmul |
| `residual_add` | hidden += residual |
| `swiglu` | silu(gate) * up |
| `rope` | Rotary Position Embedding |

### Matmul Operations

| Operation | Weight Type | Default Tile Rows |
|----------|----------|----------------|
| `matmul_q` | Q5_0 | 8 |
| `matmul_k` | Q5_0 | 8 |
| `matmul_v` | Q8_0 | 64 |
| `matmul_output` | Q5_0 | 8 |
| `matmul_gate` | Q5_0 | 8 |
| `matmul_up` | Q5_0 | 8 |
| `matmul_down` | Q6_K/Q4_K | 32/56 |
| `token_embedding` | Q8_0 | 64 |
| `lm_head` | Q8_0 | 64 |

## Tensor Type Constants

Used in descriptor `tensor_type` field:

| Type | Value | Description |
|------|-------|-------------|
| `INT16` | 1 | Standard INT16×INT16 |
| `Q5_0` | 6 | 5-bit quantization |
| `Q8_0` | 8 | 8-bit quantization |
| `Q4_K` | 12 | 4-bit K-quantization |
| `Q6_K` | 14 | 6-bit K-quantization |
| `CPU_OP` | 15 | CPU-handled operation |

## Tile Row Sizes

Different cores process different tile sizes:

| Type | Tile Rows | Tile Cols | Bytes/Tile |
|------|---------|----------|-----------|
| Q8_0 | 64 | 896 | 4100 |
| Q5_0 | 8 | 896 | 4928 |
| Q6_K | 32 | 256 | 6720 |
| Q4_K | 56 | 256 | 8064 |
| INT16 | 64 | 64 | 8192 |

## Descriptor Binary Format

Each descriptor is 32 bytes:

| Offset | Size | Field |
|--------|------|-------|
| 0 | 4 | next_desc_addr |
| 4 | 4 | weight_addr |
| 8 | 4 | act_addr |
| 12 | 4 | result_addr |
| 16 | 2 | num_tiles |
| 18 | 2 | tile_bytes |
| 20 | 1 | tensor_type |
| 21 | 1 | tile_res_rows |
| 22 | 1 | flags |
| 23 | 2 | act_total_bytes |
| 25 | 1 | num_col_groups |
| 26 | 6 | reserved |

## Adding New Models

### Step 1: Create Stage Files

Create `stages/embedding.json`:
```json
{
  "stage": "embedding",
  "ops": [
    {"name": "token_embedding", "weight": "...", "in": "...", "out": "...", "quant_type": "Q8_0"}
  ]
}
```

### Step 2: Create Layer Files

For each layer, create `layers/layer_XX.json` with the 12 standard operations (adjust quantization as needed).

### Step 3: Create Model Config

Reference all stages in `model.json`:

```json
{
  "model": "my-model",
  "num_layers": N,
  "hidden_dim": H,
  "stages": [
    {"name": "embedding", "ops": [...]},
    {"name": "layer_00", "ref": "layers/layer_00.json"},
    // ... more layers ...
    {"name": "logits", "ops": [...]}
  ]
}
```

### Step 4: Compile

```bash
python3 compile_pipeline.py model.json -o firmware.bin
```

## Moving Ops Between CPU and FPGA

To move an op from CPU to FPGA:

1. **Change name**: Ensure op name follows convention (`matmul_*`, `token_*`, etc.)
2. **Set quant_type**: Add `"quant_type": "Q5_0"` (or appropriate type)
3. **Remove from CPU_OP**: Remove from CPU operations list

To move an op from FPGA to CPU:

1. **Rename**: Use standard CPU op name (`rms_norm`, `softmax_attention`, etc.)
2. **Remove quant_type**: Remove the field
3. **CPU handles**: Add handler in C++ simulation

## Advanced: Custom Tile Sizes

Modify `OP_TILE_ROWS` in `compile_pipeline.py`:

```python
OP_TILE_ROWS = {
    "Q8_0": 64,    # Change if needed
    "Q5_0": 8,
    "Q6_K": 32,
    "Q4_K": 56,
    "INT16": 64,
    "CPU_OP": 0,
}
```

## Testing

### Verify Binary

```bash
python3 -c "
import struct
with open('firmware.bin', 'rb') as f:
    data = f.read()
for i in range(0, len(data), 32):
    d = data[i:i+32]
    _, w, a, r, nt, tb, tt, tr, fl, at, ng, _ = struct.unpack('<IIIIHHBBBHB6s', d)
    print(f'{i//32:3d}: type={tt:2d} rows={tr:2d}')
"
```

### Count by Type

```bash
python3 -c "
import struct, collections
with open('firmware.bin', 'rb') as f:
    data = f.read()
t = collections.Counter()
for i in range(0, len(data), 32):
    tt = data[i+20]
    t[tt] += 1
for k,v in sorted(t.items()):
    print(f'type {k}: {v}')
"
```

## Future Enhancements

- [ ] YAML support for configs
- [ ] Config validator (check address continuity)
- [ ] Visualizer (DOT graph export)
- [ ] Multi-head attention variants
- [ ] RoPE operator
- [ ] Graph optimizer (fold CPU_OPs, reorder independent ops)
- [ ] Support for Flash Attention

## License

MIT