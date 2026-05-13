# Analysis

## Checkpoint Files in /tmp/cpp_inter/ (all layer 0, single token)

| File | Size | Dim | Content |
|------|------|-----|---------|
| attn_norm_out.bin | 3,584B | 896 | RMS norm of embedding |
| q_after_bias.bin | 3,584B | 896 | Q = Wq @ attn_norm + bias |
| k_after_bias.bin | 512B | 128 | K = Wk^T @ attn_norm + bias |
| v_after_bias.bin | 512B | 128 | V = Wv^T @ attn_norm + bias |
| context.bin | 3,584B | 896 | attention output (== V for single-token) |
| attn_out.bin | 3,584B | 896 | Wo @ context |
| after_attn_res.bin | 3,584B | 896 | residual: hidden + attn_out |
| ffn_norm_out.bin | 3,584B | 896 | RMS norm of residual |
| gate.bin | 19,456B | 4864 | Wgate^T @ ffn_norm |
| up.bin | 19,456B | 4864 | Wup^T @ ffn_norm |
| gate_x_up.bin | 19,456B | 4864 | SiLU(gate) * up |
| ffn_out.bin | 3,584B | 896 | Wdown^T @ (SiLU(gate)*up) |
| final.bin | 3,584B | 896 | residual + ffn_out |

Also `/tmp/cpp_layer_{0..24}.bin` — layer outputs (layer 0 = embedding, layers 1-24 = after each transformer layer)

## Model Dimensions

| Parameter | Value |
|-----------|-------|
| hidden_size | 896 |
| intermediate_size | 4864 |
| vocab_size | 151936 |
| num_layers | 24 |
| num_attention_heads | 14 |
| num_kv_heads | 2 |
| head_dim | 64 |
| rope_theta | 1,000,000 |
| rms_norm_eps (config) | 1e-6 |
| tie_word_embeddings | true |

## C++ Inference Flow

1. **Embedding**: process_embedding reads from token_embd.weight (Q8_0, physical layout [151936×896], indexed as token_id * 896 + feature)
2. **RMS Norm**: rms_norm(x, w) = x * w / (sqrt(mean(x²)) + eps) — BUG: wrong formula
3. **QKV**: Q via matmul (W_q @ x), K/V via matmul_transposed (W_k^T @ x, W_v^T @ x), biases added
4. **RoPE**: Applied per-head, per-pair with theta = 1/rope_base^(d/head_dim), same in Python
5. **Attention**: GQA, softmax by subtracting max, exponentiating, normalizing
6. **Output projection**: W_o @ context via matmul
7. **Residual**: hidden += attn_out
8. **FFN**: RMS norm → W_gate^T, W_up^T → SiLU(gate) * up → W_down^T → residual
9. **Output**: RMS norm → logits = hidden @ token_embd weight (tied)

## Bugs Found

### BUG 1: RMS Norm formula (CRITICAL) — tmac_gguf.cpp:166

```c
// C++ (WRONG)
float r = rms(x, n);  // = sqrt(mean(x²))
o[i] = x[i] * w[i] / (r + 1e-5f);
```

```cpp
// ggml (CORRECT) — ops.cpp:3763-3764
const float mean = sum/n;
const float scale = 1.0f/sqrtf(mean + eps);
// then y[i] = x[i] * scale * w[i];
```

The formula x*w/(sqrt(mean)+eps) vs x*w/sqrt(mean+eps) gives ~2.6% error at the first RMS norm call (embedding std is very small at 0.0136). This cascades through all 24 layers.

### BUG 2: Wrong epsilon — tmac_gguf.cpp:166

C++ hardcodes `1e-5f` but the Qwen2 config says `rms_norm_eps: 1e-6`. llama.cpp loads this dynamically from GGUF metadata via `ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, ...)`.

### BUG 3: Python reference also has wrong epsilon — py_tmac_vec.py:204

The Python reference uses `eps = np.float32(1e-5)` instead of `1e-6`. The RMS norm formula is correct (sqrt(mean + eps)), but the eps value is wrong.

## Python vs C++ Approach Differences

The overall algorithm is the same (both implement GQA, SwiGLU FFN, RMS norm, RoPE, tied embeddings). The differences are:

- **RMS norm formula** (C++ wrong, Python correct, both wrong eps)
- **Dequantization** — both reimplement ggml quant formats; need independent verification
- **Token assignment** — must ensure both use the SAME token ID (9707 = "Hello")

## Getting Ground Truth from llama.cpp

The llama.cpp `llm_graph_context::cb` callback is called for every intermediate tensor (attn_norm, Qcur, Kcur, Vcur, ffn_norm, ffn_out, l_out, result_norm, result_output, etc.). It currently only names tensors and handles device offloads. To dump intermediates:

> Modify llama_context::graph_get_cb() in llama-context.cpp:2201 to save tensor data to files, keyed by name-il. The tensors are float* with ne[0] elements.

## Plan

### Phase 1: Fix C++ RMS norm

- [ ] Change rms_norm formula from x*w/(r+eps) to x*w/sqrt(mean(x²)+eps)
- [ ] Change eps from 1e-5f to 1e-6f (or load from config)
- [ ] Fix rms_norm with correct formula

### Phase 2: Fix Python reference epsilon

- [ ] Change eps from 1e-5 to 1e-6

### Phase 3: Get llama.cpp ground truth

- [ ] Add a --dump-intermediates flag to llama.cpp's main that modifies the cb callback to save each intermediate tensor to /tmp/llama_inter/
- [ ] Run with `echo "Hello" | ./main ... --prompt "Hello"` or equivalent to get single-token (token 9707) intermediates

### Phase 4: Compare and iterate

- [ ] Compare each C++ intermediate against Python (same formula, different eps)
- [ ] Compare C++ with corrected formula + eps against llama.cpp ground truth
- [ ] Find the first layer where outputs diverge after fixing RMS norm

---

Would you like me to proceed with implementing any of these phases?
