#!/usr/bin/env python3
"""
Verify C++ inference layer by layer - FULL ATTENTION, ALL LAYERS (Optimized)
"""
import sys
import numpy as np

def f16_to_f32(f16):
    sign = (f16 >> 15) & 1
    exp = (f16 >> 10) & 0x1F
    mant = f16 & 0x3FF
    if exp == 0:
        if mant == 0: return 0.0
        return (-1 if sign else 1) * (mant / 1024.0) * (2 ** -14)
    elif exp == 31:
        return float('nan') if mant else (float('-inf') if sign else float('inf'))
    else:
        return (-1 if sign else 1) * (1.0 + mant / 1024.0) * (2 ** (exp - 15))

def dequant_q5_0(data, idx):
    block = idx // 32
    offset = idx % 32
    base = block * 22
    d_raw = int.from_bytes(data[base:base+2], 'little')
    d = f16_to_f32(d_raw)
    qh = int.from_bytes(data[base+2:base+6], 'little')
    qs_byte = data[base + 6 + offset // 2]
    ql = (qs_byte & 0xF) if offset % 2 == 0 else (qs_byte >> 4)
    qh_bit = (qh >> offset) & 1
    q = ((qh_bit << 4) | ql) - 16
    return d * q

def dequant_q8_0(data, idx):
    block = idx // 32
    offset = idx % 32
    base = block * 34
    scale_raw = int.from_bytes(data[base:base+2], 'little')
    scale = f16_to_f32(scale_raw)
    val = data[base + 2 + offset]
    if val > 127: val -= 256
    return val * scale

def read_tensor_fast(data, rows, cols, tensor_type):
    if tensor_type == 0:
        return np.frombuffer(data, dtype=np.float32).reshape(rows, cols)
    elif tensor_type == 6:
        result = np.zeros((rows, cols), dtype=np.float32)
        for i in range(rows):
            for j in range(cols):
                result[i, j] = dequant_q5_0(data, i * cols + j)
        return result
    elif tensor_type == 8:
        result = np.zeros((rows, cols), dtype=np.float32)
        for i in range(rows):
            for j in range(cols):
                result[i, j] = dequant_q8_0(data, i * cols + j)
        return result
    elif tensor_type == 14:
        QK_K = 256
        result = np.zeros((rows, cols), dtype=np.float32)
        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                block = idx // QK_K
                offset = idx % QK_K
                base = block * 210
                d_raw = int.from_bytes(data[base+208:base+210], 'little')
                d = f16_to_f32(d_raw)
                ql_nibble = (data[base + offset // 2] & 0xF) if offset % 2 == 0 else (data[base + offset // 2] >> 4)
                qh_nibble = (data[base + 128 + offset // 4] >> ((offset % 4) * 2)) & 0x3
                q = ((qh_nibble << 4) | ql_nibble) - 32
                scale_idx = offset // 16
                scale = data[base + 192 + scale_idx]
                if scale > 127: scale -= 256
                result[i, j] = d * scale * q
        return result
    return np.zeros((rows, cols))

def rms_norm(x, w):
    r = np.sqrt(np.mean(x**2))
    return x * w / (r + 1e-5)

def silu(x):
    return x / (1 + np.exp(-x))

def forward_layer(hidden, layer, tensors, cache, HIDDEN_DIM, INTER_DIM, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM):
    cache_key = f'layer{layer}'
    if cache_key not in cache:
        t_q = tensors[f'blk.{layer}.attn_q.weight']
        t_k = tensors[f'blk.{layer}.attn_k.weight']
        t_v = tensors[f'blk.{layer}.attn_v.weight']
        t_o = tensors[f'blk.{layer}.attn_output.weight']
        t_fn = tensors[f'blk.{layer}.ffn_norm.weight']
        t_gate = tensors[f'blk.{layer}.ffn_gate.weight']
        t_up = tensors[f'blk.{layer}.ffn_up.weight']
        t_down = tensors[f'blk.{layer}.ffn_down.weight']
        t_an = tensors[f'blk.{layer}.attn_norm.weight']

        cache[cache_key] = {
            'attn_q': read_tensor_fast(t_q['data'], t_q['rows'], t_q['cols'], t_q['type']),
            'attn_k': read_tensor_fast(t_k['data'], t_k['rows'], t_k['cols'], t_k['type']),
            'attn_v': read_tensor_fast(t_v['data'], t_v['rows'], t_v['cols'], t_v['type']),
            'attn_o': read_tensor_fast(t_o['data'], t_o['rows'], t_o['cols'], t_o['type']),
            'ffn_norm': read_tensor_fast(t_fn['data'], t_fn['rows'], t_fn['cols'], t_fn['type']),
            'ffn_gate': read_tensor_fast(t_gate['data'], t_gate['rows'], t_gate['cols'], t_gate['type']),
            'ffn_up': read_tensor_fast(t_up['data'], t_up['rows'], t_up['cols'], t_up['type']),
            'ffn_down': read_tensor_fast(t_down['data'], t_down['rows'], t_down['cols'], t_down['type']),
            'attn_norm': read_tensor_fast(t_an['data'], t_an['rows'], t_an['cols'], t_an['type']),
        }
        print(f"  [Cache loaded for layer {layer}]", flush=True)

    w = cache[cache_key]

    # Attention norm
    hn = rms_norm(hidden, w['attn_norm'])

    # QKV
    Q = hn @ w['attn_q'].T
    K = hn @ w['attn_k'].T
    V = hn @ w['attn_v'].T

    # GQA attention (single token)
    context = np.zeros(HIDDEN_DIM)
    q_per_kv = NUM_HEADS // NUM_KV_HEADS
    for qh in range(NUM_HEADS):
        kv = qh // q_per_kv
        qh_data = Q[qh*HEAD_DIM:(qh+1)*HEAD_DIM]
        kh_data = K[kv*HEAD_DIM:(kv+1)*HEAD_DIM]
        vh_data = V[kv*HEAD_DIM:(kv+1)*HEAD_DIM]
        score = np.dot(qh_data, kh_data) / np.sqrt(HEAD_DIM)
        softmax_w = 1.0  # single token
        context[qh*HEAD_DIM:(qh+1)*HEAD_DIM] = softmax_w * vh_data

    # Attention output projection
    attn_out = context @ w['attn_o'].T

    # Residual
    hidden = hidden + attn_out

    # FFN norm
    hn2 = rms_norm(hidden, w['ffn_norm'])

    # FFN
    gate = silu(hn2 @ w['ffn_gate'].T) * (hn2 @ w['ffn_up'].T)
    ffn_out = gate @ w['ffn_down'].T

    # Final residual
    hidden = hidden + ffn_out
    return hidden

def main():
    if len(sys.argv) < 2:
        print("Usage: verify_layers_fast.py <model.tmac> [max_layers]")
        return 1

    model_path = sys.argv[1]
    max_layers = int(sys.argv[2]) if len(sys.argv) > 2 else 24

    with open(model_path, 'rb') as f:
        magic = f.read(4)
        n_tensors = int.from_bytes(f.read(8), 'little')
        tensors = {}
        for i in range(n_tensors):
            name_len = int.from_bytes(f.read(8), 'little')
            name = f.read(name_len).decode('utf-8')
            rows = int.from_bytes(f.read(8), 'little')
            cols = int.from_bytes(f.read(8), 'little')
            typ = int.from_bytes(f.read(4), 'little')
            n_bytes = int.from_bytes(f.read(8), 'little')
            tensors[name] = {'rows': rows, 'cols': cols, 'type': typ, 'data': f.read(n_bytes)}

    print(f"Loaded {len(tensors)} tensors")

    token_id = 100
    HIDDEN_DIM = 896
    INTER_DIM = 4864
    NUM_HEADS = 14
    NUM_KV_HEADS = 2
    HEAD_DIM = HIDDEN_DIM // NUM_HEADS

    cache = {}

    # Get embedding
    emb = tensors['token_embd.weight']
    hidden = np.array([read_tensor_fast(emb['data'], emb['rows'], emb['cols'], emb['type'])[:, token_id]]).flatten()
    print(f"\nInitial hidden (token={token_id}): {hidden[:3]}")

    for layer in range(max_layers):
        print(f"Layer {layer}: ", end="", flush=True)
        hidden = forward_layer(hidden, layer, tensors, cache, HIDDEN_DIM, INTER_DIM, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM)
        print(f"{hidden[0]:.6f},{hidden[1]:.6f},{hidden[2]:.6f}")

    return 0

if __name__ == '__main__':
    sys.exit(main())