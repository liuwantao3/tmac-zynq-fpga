# Model Files

## Source GGUF
- `qwen2-0_5b-instruct-q4_k_m.gguf` (~392 MB)
- Source: HuggingFace Qwen2-0.5B-Instruct, quantized to q4_k_m
- Format: GGUF (llama.cpp compatible)
- Located at: `/Users/arctic/Downloads/qwen2-0_5b-instruct-q4_k_m.gguf` (original)

## TMAC Format
The TMAC model is generated from GGUF via `scripts/extract_tmac.py`:
- Located at: `/tmp/model.tmac` (373.7 MB, 290 tensors)
- Contains raw quantized weights (same bit patterns as GGUF) in flat binary format

## Conversion
```bash
python3 scripts/extract_tmac.py
# Reads from hardcoded GGUF path, writes /tmp/model.tmac
```

## Weight Count by Quantization
Verified from the GGUF header (290 tensors total):

| Type | Count | Tensors |
|------|-------|---------|
| Q5_0 | 132 | `attn_q/k/output.weight` (24 each), `ffn_gate/up.weight` (24 each), `attn_v.weight` (12 layers) |
| Q8_0 | 13 | `token_embd.weight` + `attn_v.weight` (12 layers) |
| Q6_K | 12 | `blk.*.ffn_down.weight` (even layers) |
| Q4_K | 12 | `blk.*.ffn_down.weight` (odd layers) |
| F32 | 121 | `attn_q/k/v.bias` (24 each), `attn_norm/ffn_norm.weight` (24 each), `output_norm.weight` |
