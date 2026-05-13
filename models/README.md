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
| Type | Count | Tensors |
|------|-------|---------|
| Q5_0 | 132 | Most weight matrices |
| Q6_K | 12 | `blk.*.ffn_down.weight` |
| Q8_0 | 2 | `token_embd.weight`, `attn_v.weight` |
| Q4_K | rest | Remaining weight matrices |
| F32 | few | Norm/bias (`*.weight`, `*.bias`) |
