#!/usr/bin/env python3
"""
Python inference using GGUF reader - for generating ground truth.
Records intermediate layer outputs for comparison with C++ TMAC inference.
"""

import sys
import numpy as np
from gguf import GGUFReader

# Config - Qwen2-0.5B
HIDDEN_DIM = 896
INTER_DIM = 4864
VOCAB_SIZE = 151936
NUM_LAYERS = 24
NUM_HEADS = 14
HEAD_DIM = 64
NUM_KV_HEADS = 2
K_DIM = NUM_KV_HEADS * HEAD_DIM
V_DIM = K_DIM

def f16_to_f32(f16):
    """Convert float16 to float32"""
    sign = (f16 >> 15) & 1
    exp = (f16 >> 10) & 0x1F
    mant = f16 & 0x3FF

    if exp == 0:
        if mant == 0:
            return 0.0
        return (mant / 1024.0) * (2 ** -14)
    elif exp == 31:
        return float('-inf') if sign else float('inf')
    else:
        return (1 + mant / 1024.0) * (2 ** (exp - 15))

def dequant_q8_0(data, idx):
    """Dequantize Q8_0 format"""
    block = idx // 32
    offset = idx % 32
    block_offset = block * 34
    scale_raw = int(data[block_offset]) | (int(data[block_offset + 1]) << 8)
    scale = f16_to_f32(scale_raw)
    q = int(data[block_offset + 2 + offset])
    if q > 127:
        q -= 256
    return scale * q

def read_tensor(tensor_info, data, row, col):
    """Read a value from a tensor"""
    if tensor_info['type'] == 0:  # F32
        idx = row * tensor_info['cols'] + col
        return float(np.frombuffer(data[idx*4:(idx+1)*4], dtype=np.float32)[0])
    elif tensor_info['type'] == 1:  # F16
        idx = row * tensor_info['cols'] + col
        raw = int(data[idx*2]) | (int(data[idx*2+1]) << 8)
        return f16_to_f32(raw)
    elif tensor_info['type'] == 8:  # Q8_0
        idx = row * tensor_info['cols'] + col
        return dequant_q8_0(data, idx)
    return 0.0

def read_embedding(tensor_info, emb_data, token_id, feature):
    """Read embedding - GGUF shape is [hidden, vocab]"""
    # Element [feature, token_id] is at index feature * vocab + token_id
    idx = feature * tensor_info['cols'] + token_id
    return dequant_q8_0(emb_data, idx)

def rms_norm(x, weight, eps=1e-5):
    """RMS normalization"""
    rms = np.sqrt(np.mean(x ** 2) + eps)
    return x * weight / rms

def silu(x):
    """SiLU activation"""
    return x / (1.0 + np.exp(-x))

def matmul(A, x, rows, cols, tensor_info, data):
    """Matrix multiplication - A is [rows x cols], x is [cols]"""
    y = np.zeros(rows)
    for i in range(rows):
        for j in range(cols):
            y[i] += read_tensor(tensor_info, data, i, j) * x[j]
    return y

def matmul_transposed(A, x, rows, cols, tensor_info, data):
    """Matrix multiplication with transposed A - A is [cols x rows]"""
    # A_stored[j][i] corresponds to A_conceptual[i][j]
    y = np.zeros(rows)
    for i in range(rows):
        for j in range(cols):
            y[i] += read_tensor(tensor_info, data, j, i) * x[j]
    return y

def apply_rope(q, k, pos):
    """Apply rotary positional encoding"""
    for h in range(NUM_HEADS):
        for d in range(0, HEAD_DIM, 2):
            theta = 1.0 / (10000.0 ** (d / HEAD_DIM))
            freq = pos * theta
            cos_val = np.cos(freq)
            sin_val = np.sin(freq)
            idx = h * HEAD_DIM + d
            q0, q1 = q[idx], q[idx + 1]
            q[idx] = q0 * cos_val - q1 * sin_val
            q[idx + 1] = q0 * sin_val + q1 * cos_val

    for h in range(NUM_KV_HEADS):
        for d in range(0, HEAD_DIM, 2):
            theta = 1.0 / (10000.0 ** (d / HEAD_DIM))
            freq = pos * theta
            cos_val = np.cos(freq)
            sin_val = np.sin(freq)
            idx = h * HEAD_DIM + d
            k0, k1 = k[idx], k[idx + 1]
            k[idx] = k0 * cos_val - k1 * sin_val
            k[idx + 1] = k0 * sin_val + k1 * cos_val

def forward_layer(hidden, layer, pos, tensors, layer_data, k_cache, v_cache):
    """Forward pass through one transformer layer"""
    original_hidden = hidden.copy()

    # Attention norm
    attn_norm = tensors[f'blk.{layer}.attn_norm.weight']
    attn_norm_out = rms_norm(hidden, layer_data[f'blk.{layer}.attn_norm.weight'])

    # Q, K, V projections
    q = matmul(tensors[f'blk.{layer}.attn_q.weight'], attn_norm_out,
               HIDDEN_DIM, HIDDEN_DIM, tensors[f'blk.{layer}.attn_q.weight'], layer_data[f'blk.{layer}.attn_q.weight'])
    k = matmul(tensors[f'blk.{layer}.attn_k.weight'], attn_norm_out,
                K_DIM, HIDDEN_DIM, tensors[f'blk.{layer}.attn_k.weight'], layer_data[f'blk.{layer}.attn_k.weight'])
    v = matmul(tensors[f'blk.{layer}.attn_v.weight'], attn_norm_out,
                V_DIM, HIDDEN_DIM, tensors[f'blk.{layer}.attn_v.weight'], layer_data[f'blk.{layer}.attn_v.weight'])

    # Apply RoPE
    apply_rope(q, k, pos)

    # Store in cache
    k_cache[pos] = k.copy()
    v_cache[pos] = v.copy()

    # GQA Attention
    context = np.zeros(HIDDEN_DIM)
    q_per_kv = NUM_HEADS // NUM_KV_HEADS

    for qh in range(NUM_HEADS):
        kv = qh // q_per_kv
        qh_data = q[qh * HEAD_DIM:(qh + 1) * HEAD_DIM]
        ctx_h = context[qh * HEAD_DIM:(qh + 1) * HEAD_DIM]

        # Compute attention scores
        scores = []
        for p in range(pos + 1):
            k_cached = k_cache[p][kv * HEAD_DIM:(kv + 1) * HEAD_DIM]
            score = np.dot(qh_data, k_cached) / np.sqrt(HEAD_DIM)
            scores.append(score)

        # Softmax
        scores = np.array(scores)
        scores -= np.max(scores)
        exp_scores = np.exp(scores)
        exp_scores /= np.sum(exp_scores)

        # Compute context
        for d in range(HEAD_DIM):
            weighted_sum = 0.0
            for p in range(pos + 1):
                v_cached = v_cache[p][kv * HEAD_DIM:(kv + 1) * HEAD_DIM]
                weighted_sum += exp_scores[p] * v_cached[d]
            ctx_h[d] = weighted_sum

    # Attention output projection
    attn_out = matmul(tensors[f'blk.{layer}.attn_output.weight'], context,
                       HIDDEN_DIM, HIDDEN_DIM, tensors[f'blk.{layer}.attn_output.weight'], layer_data[f'blk.{layer}.attn_output.weight'])

    # Residual
    hidden = original_hidden + attn_out

    # FFN norm
    ffn_norm_out = rms_norm(hidden, layer_data[f'blk.{layer}.ffn_norm.weight'])

    # FFN gate and up projections
    gate = matmul_transposed(tensors[f'blk.{layer}.ffn_gate.weight'], ffn_norm_out,
                             INTER_DIM, HIDDEN_DIM, tensors[f'blk.{layer}.ffn_gate.weight'], layer_data[f'blk.{layer}.ffn_gate.weight'])
    up = matmul_transposed(tensors[f'blk.{layer}.ffn_up.weight'], ffn_norm_out,
                            INTER_DIM, HIDDEN_DIM, tensors[f'blk.{layer}.ffn_up.weight'], layer_data[f'blk.{layer}.ffn_up.weight'])

    # SwiGLU
    gate = silu(gate)
    gate = gate * up

    # FFN down
    ffn_out = matmul_transposed(tensors[f'blk.{layer}.ffn_down.weight'], gate,
                                 HIDDEN_DIM, INTER_DIM, tensors[f'blk.{layer}.ffn_down.weight'], layer_data[f'blk.{layer}.ffn_down.weight'])

    # Final residual
    hidden = hidden + ffn_out

    return hidden

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 gguf_inference.py <model.gguf> <token_id> [token_id2 ...]")
        sys.exit(1)

    model_path = sys.argv[1]
    token_ids = [int(t) for t in sys.argv[2:]]

    print(f"Loading model from {model_path}...")
    reader = GGUFReader(model_path)

    # Load tensor metadata
    tensors = {}
    for t in reader.tensors:
        tensors[t.name] = {
            'rows': t.shape[0] if len(t.shape) > 0 else 1,
            'cols': t.shape[1] if len(t.shape) > 1 else t.shape[0],
            'type': t.tensor_type.value,
            'offset': t.data_offset,
            'n_bytes': t.n_bytes
        }

    # Load embedding data
    emb_t = tensors['token_embd.weight']
    emb_data = reader.data[emb_t['offset']:emb_t['offset'] + emb_t['n_bytes']]

    # Load layer weights into memory (for speed)
    print("Loading layer weights...")
    layer_data = {}
    for layer in range(NUM_LAYERS):
        for name in [f'blk.{layer}.attn_norm.weight',
                     f'blk.{layer}.attn_q.weight', f'blk.{layer}.attn_k.weight', f'blk.{layer}.attn_v.weight',
                     f'blk.{layer}.attn_output.weight',
                     f'blk.{layer}.ffn_norm.weight',
                     f'blk.{layer}.ffn_gate.weight', f'blk.{layer}.ffn_up.weight', f'blk.{layer}.ffn_down.weight']:
            t = tensors[name]
            raw_data = reader.data[t['offset']:t['offset'] + t['n_bytes']]
            if t['type'] == 0:  # F32
                layer_data[name] = np.frombuffer(raw_data.tobytes(), dtype=np.float32)
            elif t['type'] == 1:  # F16
                raw_uint16 = np.frombuffer(raw_data.tobytes(), dtype=np.uint16)
                layer_data[name] = np.array([f16_to_f32(x) for x in raw_uint16], dtype=np.float32)
            else:
                layer_data[name] = raw_data  # Keep quantized data for later dequantization

    # KV cache
    k_cache = [np.zeros(K_DIM) for _ in range(256)]
    v_cache = [np.zeros(V_DIM) for _ in range(256)]

    # Get embedding for first token
    hidden = np.zeros(HIDDEN_DIM)
    for i in range(HIDDEN_DIM):
        hidden[i] = read_embedding(emb_t, emb_data, token_ids[0], i)

    print(f"\n=== Layer 0 Input Embedding ===")
    print(f"hidden[0:5] = {hidden[0:5]}")
    print(f"mean = {hidden.mean():.6f}, std = {hidden.std():.6f}")

    # Save embedding as layer 0 output
    layer_outputs = [hidden.copy()]

    # Forward through all layers
    for layer in range(NUM_LAYERS):
        hidden = forward_layer(hidden, layer, 0, tensors, layer_data, k_cache, v_cache)
        print(f"\n=== Layer {layer} Output ===")
        print(f"hidden[0:5] = {hidden[0:5]}")
        print(f"mean = {hidden.mean():.6f}, std = {hidden.std():.6f}")
        layer_outputs.append(hidden.copy())

    # Final output norm
    output_norm = layer_data['output_norm.weight']
    hidden_rms = rms_norm(hidden, output_norm)
    print(f"\n=== Final Output (after output_norm) ===")
    print(f"hidden[0:5] = {hidden_rms[0:5]}")
    print(f"mean = {hidden_rms.mean():.6f}, std = {hidden_rms.std():.6f}")

    # Compute logits
    print(f"\n=== Computing Logits ===")
    logits = np.zeros(VOCAB_SIZE)
    for v in range(VOCAB_SIZE):
        for h in range(HIDDEN_DIM):
            logits[v] += hidden_rms[h] * read_embedding(emb_t, emb_data, v, h)

    # Top 10 logits
    top_indices = np.argsort(logits)[-10:][::-1]
    print("Top 10 logits:")
    for idx in top_indices:
        print(f"  token {idx}: {logits[idx]:.3f}")

    # Save all layer outputs for comparison with C++
    print("\n=== Saving Layer Outputs ===")
    for layer, output in enumerate(layer_outputs):
        filename = f"/tmp/py_layer_{layer}_emb.bin"
        output.tofile(filename)
        print(f"Layer {layer}: saved to {filename}")

    # Also save final hidden state
    hidden_rms.tofile("/tmp/py_final_hidden.bin")
    print(f"Final hidden: saved to /tmp/py_final_hidden.bin")

    # Save logits
    logits.tofile("/tmp/py_logits.bin")
    print(f"Logits: saved to /tmp/py_logits.bin")

if __name__ == "__main__":
    main()