#!/usr/bin/env python3
"""
Python inference that mirrors C++ TMAC - for generating ground truth layer outputs.
Loads TMAC format (same as C++ TMAC) and runs full model inference.
"""

import sys
import numpy as np
import struct

# Config - Qwen2-0.5B
HIDDEN_DIM = 896
INTER_DIM = 4864
VOCAB_SIZE = 151936
NUM_LAYERS = 24
NUM_HEADS = 14
HEAD_DIM = 64
NUM_KV_HEADS = 2
K_DIM = NUM_KV_HEADS * HEAD_DIM
MAX_SEQ_LEN = 256

def f16_to_f32(f16):
    sign = (f16 >> 15) & 1
    exp = (f16 >> 10) & 0x1F
    mant = f16 & 0x3FF
    if exp == 0:
        if mant == 0: return 0.0
        return (mant / 1024.0) * (2 ** -14)
    elif exp == 31:
        return float('-inf') if sign else float('inf')
    return (1 + mant / 1024.0) * (2 ** (exp - 15))

def dequant_q8_0(data, idx):
    block = idx // 32
    offset = idx % 32
    block_offset = block * 34
    scale_raw = int(data[block_offset]) | (int(data[block_offset + 1]) << 8)
    scale = f16_to_f32(scale_raw)
    q = int(data[block_offset + 2 + offset])
    if q > 127: q -= 256
    return scale * q

def load_tmac(filepath):
    """Load TMAC model"""
    with open(filepath, 'rb') as f:
        magic = f.read(4)
        n_tensors = struct.unpack('<Q', f.read(8))[0]

        tensors = {}
        for _ in range(n_tensors):
            name_len = struct.unpack('<Q', f.read(8))[0]
            name = f.read(name_len).decode('utf-8')
            rows = struct.unpack('<Q', f.read(8))[0]
            cols = struct.unpack('<Q', f.read(8))[0]
            tensor_type = struct.unpack('<I', f.read(4))[0]
            n_bytes = struct.unpack('<Q', f.read(8))[0]

            data = f.read(n_bytes)
            tensors[name] = {
                'rows': rows,
                'cols': cols,
                'type': tensor_type,
                'data': data
            }
            print(f"Loaded: {name} [{rows}x{cols}] type={tensor_type}")
    return tensors

def read_tensor(tensors, name, row, col):
    t = tensors[name]
    idx = row * t['cols'] + col
    if t['type'] == 0:  # F32
        return struct.unpack('<f', t['data'][idx*4:(idx+1)*4])[0]
    elif t['type'] == 8:  # Q8_0
        return dequant_q8_0(t['data'], idx)
    return 0.0

def read_embedding(tensors, token_id, feature):
    """Read embedding - stored as [hidden, vocab], so [feature][token_id]"""
    t = tensors['token_embd.weight']
    idx = feature * t['cols'] + token_id
    return dequant_q8_0(t['data'], idx)

def rms_norm(x, weight):
    eps = 1e-5
    rms = np.sqrt(np.mean(x**2) + eps)
    return x * weight / rms

def silu(x):
    return x / (1.0 + np.exp(-x))

def matmul_vec(tensors, name, x, rows, cols):
    """Matrix-vector multiply: A @ x where A is [rows x cols]"""
    t = tensors[name]
    y = np.zeros(rows)
    for i in range(rows):
        for j in range(cols):
            y[i] += read_tensor(tensors, name, i, j) * x[j]
    return y

def matmul_vec_transposed(tensors, name, x, rows, cols):
    """Matrix-vector multiply with transposed A: A.T @ x where A is [cols x rows]"""
    t = tensors[name]
    y = np.zeros(rows)
    for i in range(rows):
        for j in range(cols):
            # Transposed: A[j][i] instead of A[i][j]
            y[i] += read_tensor(tensors, name, j, i) * x[j]
    return y

def apply_rope(q, k, pos):
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

def forward_layer(hidden, layer, pos, tensors, k_cache, v_cache):
    original = hidden.copy()

    # Attention norm
    attn_norm_out = rms_norm(hidden, np.array([read_tensor(tensors, f'blk.{layer}.attn_norm.weight', i, 0) for i in range(HIDDEN_DIM)]))

    # Q, K, V
    q = matmul_vec(tensors, f'blk.{layer}.attn_q.weight', attn_norm_out, HIDDEN_DIM, HIDDEN_DIM)
    k = matmul_vec(tensors, f'blk.{layer}.attn_k.weight', attn_norm_out, K_DIM, HIDDEN_DIM)
    v = matmul_vec(tensors, f'blk.{layer}.attn_v.weight', attn_norm_out, K_DIM, HIDDEN_DIM)

    apply_rope(q, k, pos)

    k_cache[pos] = k.copy()
    v_cache[pos] = v.copy()

    # GQA Attention
    context = np.zeros(HIDDEN_DIM)
    q_per_kv = NUM_HEADS // NUM_KV_HEADS

    for qh in range(NUM_HEADS):
        kv = qh // q_per_kv
        qh_data = q[qh * HEAD_DIM:(qh + 1) * HEAD_DIM]
        ctx_h = context[qh * HEAD_DIM:(qh + 1) * HEAD_DIM]

        scores = []
        for p in range(pos + 1):
            k_cached = k_cache[p][kv * HEAD_DIM:(kv + 1) * HEAD_DIM]
            score = np.dot(qh_data, k_cached) / np.sqrt(HEAD_DIM)
            scores.append(score)

        scores = np.array(scores)
        scores -= np.max(scores)
        exp_scores = np.exp(scores)
        exp_scores /= np.sum(exp_scores)

        for d in range(HEAD_DIM):
            weighted_sum = 0.0
            for p in range(pos + 1):
                v_cached = v_cache[p][kv * HEAD_DIM:(kv + 1) * HEAD_DIM]
                weighted_sum += exp_scores[p] * v_cached[d]
            ctx_h[d] = weighted_sum

    # Attention output
    attn_out = matmul_vec(tensors, f'blk.{layer}.attn_output.weight', context, HIDDEN_DIM, HIDDEN_DIM)
    hidden = original + attn_out
    original = hidden.copy()

    # FFN norm
    ffn_norm_out = rms_norm(hidden, np.array([read_tensor(tensors, f'blk.{layer}.ffn_norm.weight', i, 0) for i in range(HIDDEN_DIM)]))

    # FFN
    gate = matmul_vec_transposed(tensors, f'blk.{layer}.ffn_gate.weight', ffn_norm_out, INTER_DIM, HIDDEN_DIM)
    up = matmul_vec_transposed(tensors, f'blk.{layer}.ffn_up.weight', ffn_norm_out, INTER_DIM, HIDDEN_DIM)
    gate = silu(gate)
    gate = gate * up
    ffn_out = matmul_vec_transposed(tensors, f'blk.{layer}.ffn_down.weight', gate, HIDDEN_DIM, INTER_DIM)

    hidden = original + ffn_out
    return hidden

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 py_tmac_inference.py <model.tmac> [token_id]")
        sys.exit(1)

    model_path = sys.argv[1]
    token_id = int(sys.argv[2]) if len(sys.argv) > 2 else 9707

    print(f"Loading TMAC model from {model_path}...")
    tensors = load_tmac(model_path)

    # Get embedding
    print(f"\n=== EMBEDDING (token {token_id}) ===")
    hidden = np.array([read_embedding(tensors, token_id, i) for i in range(HIDDEN_DIM)])
    print(f"hidden[0:5]: {hidden[0:5]}")
    print(f"mean: {hidden.mean():.6f}, std: {hidden.std():.6f}")
    hidden.tofile('/tmp/py_emb.bin')

    # KV cache
    k_cache = [np.zeros(K_DIM) for _ in range(MAX_SEQ_LEN)]
    v_cache = [np.zeros(K_DIM) for _ in range(MAX_SEQ_LEN)]

    # Forward through all layers
    for layer in range(NUM_LAYERS):
        hidden = forward_layer(hidden, layer, 0, tensors, k_cache, v_cache)
        print(f"\n=== LAYER {layer} ===")
        print(f"hidden[0:5]: {hidden[0:5]}")
        print(f"mean: {hidden.mean():.6f}, std: {hidden.std():.6f}")
        hidden.tofile(f'/tmp/py_layer_{layer}.bin')

    print("\n=== DONE ===")
    print("Layer outputs saved to /tmp/py_layer_*.bin")

if __name__ == "__main__":
    main()