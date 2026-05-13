#!/usr/bin/env python3
"""
Vectorized Python TMAC inference - generates layer-by-layer ground truth.
Uses numpy for fast tensor operations.
"""

import sys
import numpy as np
import struct

# Config
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
        val = (mant / 1024.0) * (2 ** -14)
    elif exp == 31:
        return float('-inf') if sign else float('inf')
    else:
        val = (1 + mant / 1024.0) * (2 ** (exp - 15))
    return -val if sign else val

def dequant_q5_0(data, idx):
    block = idx // 32
    offset = idx % 32
    base = block * 22

    d_raw = data[base] | (data[base + 1] << 8)
    d = f16_to_f32(d_raw)

    qh = data[base + 2] | (data[base + 3] << 8) | (data[base + 4] << 16) | (data[base + 5] << 24)

    j = offset if offset < 16 else offset - 16
    qs_byte = data[base + 6 + j]
    ql = (qs_byte & 0xF) if (offset < 16) else (qs_byte >> 4)

    qh_bit = (qh >> offset) & 1

    q = ((qh_bit << 4) | ql) - 16

    return d * q

def dequant_q5_0_vec(data, n_elements):
    result = np.zeros(n_elements, dtype=np.float32)
    for i in range(n_elements):
        result[i] = dequant_q5_0(data, i)
    return result

def dequant_q6_k(data, idx):
    QK_K = 256
    block = idx // QK_K
    offset = idx % QK_K
    base = block * 210
    d_raw = data[base + 208] | (data[base + 209] << 8)
    d = f16_to_f32(d_raw)
    half = offset // 128
    pos = offset % 128
    l = pos % 32
    sub = pos // 32
    ql_base = base + half * 64
    qh_base = base + 128 + half * 32
    sc_base = base + 192 + half * 8
    if sub == 0:
        ql_byte = data[ql_base + l]
        qh_shift = 0
    elif sub == 1:
        ql_byte = data[ql_base + l + 32]
        qh_shift = 2
    elif sub == 2:
        ql_byte = data[ql_base + l]
        qh_shift = 4
    else:
        ql_byte = data[ql_base + l + 32]
        qh_shift = 6
    ql_nibble = (ql_byte & 0xF) if (sub == 0 or sub == 1) else (ql_byte >> 4) & 0xF
    qh_bits = (data[qh_base + l] >> qh_shift) & 0x3
    q = ((qh_bits << 4) | ql_nibble) - 32
    is_ = l // 16
    scale_idx = is_ + sub * 2
    scale = data[sc_base + scale_idx]
    if scale > 127: scale -= 256
    return d * scale * q

def dequant_q6_k_vec(data, n_elements):
    result = np.zeros(n_elements, dtype=np.float32)
    for i in range(n_elements):
        result[i] = dequant_q6_k(data, i)
    return result

def get_scale_min_k4(j, scales):
    if j < 4:
        sc = scales[j] & 63
        m = scales[j + 4] & 63
    else:
        sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4)
        m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4)
    return sc, m

def dequant_q4_k(data, idx):
    QK_K = 256
    block = idx // QK_K
    offset = idx % QK_K
    base = block * 144

    d_raw = data[base + 0] | (data[base + 1] << 8)
    d = f16_to_f32(d_raw)
    dmin_raw = data[base + 2] | (data[base + 3] << 8)
    dmin = f16_to_f32(dmin_raw)

    sub_block = offset // 32
    q_pos = offset % 32
    qs_byte_idx = (sub_block // 2) * 32 + q_pos
    qs_byte = data[base + 16 + qs_byte_idx]
    q4 = (qs_byte & 0xF) if (sub_block % 2 == 0) else (qs_byte >> 4)

    scales = data[base + 4:base + 16]
    sc, m = get_scale_min_k4(sub_block, scales)
    return d * sc * q4 - dmin * m

def dequant_q4_k_vec(data, n_elements):
    result = np.zeros(n_elements, dtype=np.float32)
    for i in range(n_elements):
        result[i] = dequant_q4_k(data, i)
    return result
    result = np.zeros(n_elements, dtype=np.float32)
    for i in range(n_elements):
        result[i] = dequant_q5_0(data, i)
    return result

def dequant_q8_0_vec(data, n_elements):
    """Vectorized Q8_0 dequantization"""
    n_blocks = (n_elements + 31) // 32
    result = np.zeros(n_elements, dtype=np.float32)

    # Parse scales
    scales = np.zeros(n_blocks, dtype=np.float32)
    for i in range(n_blocks):
        scale_raw = int(data[i*34]) | (int(data[i*34+1]) << 8)
        scales[i] = f16_to_f32(scale_raw)

    # Dequantize each block
    for block_i in range(n_blocks):
        block_start = block_i * 34 + 2  # Skip 2-byte scale
        block_data = np.frombuffer(bytes(data[block_start:block_start+32]), dtype=np.int8)
        start = block_i * 32
        end = min(start + 32, n_elements)
        result[start:end] = scales[block_i] * block_data[:end-start].astype(np.float32)

    return result

def load_tmac_tensors(filepath):
    """Load all tensors from TMAC file"""
    tensors = {}
    with open(filepath, 'rb') as f:
        magic = f.read(4)
        n_tensors = struct.unpack('<Q', f.read(8))[0]

        for _ in range(n_tensors):
            name_len = struct.unpack('<Q', f.read(8))[0]
            name = f.read(name_len).decode('utf-8')
            rows = struct.unpack('<Q', f.read(8))[0]
            cols = struct.unpack('<Q', f.read(8))[0]
            tensor_type = struct.unpack('<I', f.read(4))[0]
            n_bytes = struct.unpack('<Q', f.read(8))[0]

            raw_data = f.read(n_bytes)

            if tensor_type == 0:  # F32
                arr = np.frombuffer(raw_data, dtype=np.float32).reshape(rows, cols)
                if arr.shape[1] == 1:
                    arr = arr.squeeze()
                tensors[name] = arr
            elif tensor_type == 8:  # Q8_0
                # Store raw bytes for now, dequantize on access
                tensors[name] = {'raw': raw_data, 'rows': rows, 'cols': cols, 'type': tensor_type}
            else:
                tensors[name] = {'raw': raw_data, 'rows': rows, 'cols': cols, 'type': tensor_type}

    return tensors

def get_embedding(tensors, token_id):
    """Get embedding for token - tensor layout [vocab][hidden] row-major"""
    emb = tensors['token_embd.weight']
    raw = emb['raw']
    # GGUF/GGML convention: ne[0]=hidden, ne[1]=vocab
    # ggml_get_rows extracts row token_id: data[token_id * hidden + feature]
    start = token_id * HIDDEN_DIM
    end = start + HIDDEN_DIM
    n_elements = emb['rows'] * emb['cols']
    full = dequant_q8_0_vec(raw, n_elements)
    return full[start:end].copy()

_dequant_cache = {}
def dequant_tensor(tensor_dict):
    """Dequantize a tensor dict to numpy array (cached)"""
    if isinstance(tensor_dict, np.ndarray):
        return tensor_dict
    cache_key = id(tensor_dict.get('raw', b''))
    if cache_key in _dequant_cache:
        return _dequant_cache[cache_key]
    raw = tensor_dict['raw']
    rows = tensor_dict['rows']
    cols = tensor_dict['cols']
    tensor_type = tensor_dict.get('type', 8)
    n_elements = rows * cols
    if tensor_type == 8:
        result = dequant_q8_0_vec(raw, n_elements).reshape(rows, cols)
    elif tensor_type == 6:
        result = dequant_q5_0_vec(raw, n_elements).reshape(rows, cols)
    elif tensor_type == 12:
        result = dequant_q4_k_vec(raw, n_elements).reshape(rows, cols)
    elif tensor_type == 14:
        result = dequant_q6_k_vec(raw, n_elements).reshape(rows, cols)
    else:
        raise ValueError(f"Unsupported tensor type: {tensor_type}")
    if result.ndim == 2 and result.shape[1] == 1:
        result = result.squeeze()
    _dequant_cache[cache_key] = result
    return result

def rms_norm(x, weight):
    eps = np.float32(1e-6)
    rms = np.sqrt(np.mean(x.astype(np.float32)**2, dtype=np.float32) + eps)
    return (x * weight / rms).astype(np.float32)

def silu(x):
    return (x / (np.float32(1.0) + np.exp(-x))).astype(np.float32)

def matmul_vec(A, x, transposed=False):
    """Matrix-vector multiply. A is [rows x cols]"""
    if transposed:
        return np.dot(x, A)
    else:
        return np.dot(A, x)

def apply_rope(q, k, pos):
    rope_base = 1000000.0
    for h in range(NUM_HEADS):
        for d in range(0, HEAD_DIM, 2):
            theta = 1.0 / (rope_base ** (d / HEAD_DIM))
            freq = pos * theta
            cos_val = np.cos(freq)
            sin_val = np.sin(freq)
            idx = h * HEAD_DIM + d
            q0, q1 = q[idx], q[idx + 1]
            q[idx] = q0 * cos_val - q1 * sin_val
            q[idx + 1] = q0 * sin_val + q1 * cos_val

    for h in range(NUM_KV_HEADS):
        for d in range(0, HEAD_DIM, 2):
            theta = 1.0 / (rope_base ** (d / HEAD_DIM))
            freq = pos * theta
            cos_val = np.cos(freq)
            sin_val = np.sin(freq)
            idx = h * HEAD_DIM + d
            k0, k1 = k[idx], k[idx + 1]
            k[idx] = k0 * cos_val - k1 * sin_val
            k[idx + 1] = k0 * sin_val + k1 * cos_val

def forward_layer(hidden, layer, pos, tensors, k_cache, v_cache):
    # Attention norm
    attn_norm = dequant_tensor(tensors[f'blk.{layer}.attn_norm.weight'])
    attn_norm_out = rms_norm(hidden, attn_norm)

    # Q, K, V
    q = matmul_vec(dequant_tensor(tensors[f'blk.{layer}.attn_q.weight']), attn_norm_out)
    q += dequant_tensor(tensors[f'blk.{layer}.attn_q.bias'])
    k = matmul_vec(dequant_tensor(tensors[f'blk.{layer}.attn_k.weight']), attn_norm_out, transposed=True)
    k += dequant_tensor(tensors[f'blk.{layer}.attn_k.bias'])
    v = matmul_vec(dequant_tensor(tensors[f'blk.{layer}.attn_v.weight']), attn_norm_out, transposed=True)
    v += dequant_tensor(tensors[f'blk.{layer}.attn_v.bias'])

    apply_rope(q, k, pos)

    k_cache[pos] = k.copy()
    v_cache[pos] = v.copy()

    # GQA Attention
    context = np.zeros(HIDDEN_DIM, dtype=np.float32)
    q_per_kv = NUM_HEADS // NUM_KV_HEADS

    for qh in range(NUM_HEADS):
        kv = qh // q_per_kv
        qh_data = q[qh * HEAD_DIM:(qh + 1) * HEAD_DIM]
        ctx_h = context[qh * HEAD_DIM:(qh + 1) * HEAD_DIM]

        scores = np.array([np.dot(qh_data, k_cache[p][kv*HEAD_DIM:(kv+1)*HEAD_DIM]) / np.sqrt(HEAD_DIM) for p in range(pos + 1)], dtype=np.float32)
        scores -= np.max(scores)
        exp_scores = np.exp(scores)
        exp_scores /= np.sum(exp_scores)

        for d in range(HEAD_DIM):
            ctx_h[d] = sum(exp_scores[p] * v_cache[p][kv*HEAD_DIM+d] for p in range(pos + 1))

    # Attention output
    attn_out = matmul_vec(dequant_tensor(tensors[f'blk.{layer}.attn_output.weight']), context)
    hidden = (hidden + attn_out).astype(np.float32)

    # FFN norm
    ffn_norm_out = rms_norm(hidden, dequant_tensor(tensors[f'blk.{layer}.ffn_norm.weight']))

    # FFN
    gate = silu(matmul_vec(dequant_tensor(tensors[f'blk.{layer}.ffn_gate.weight']), ffn_norm_out, transposed=True))
    up = matmul_vec(dequant_tensor(tensors[f'blk.{layer}.ffn_up.weight']), ffn_norm_out, transposed=True)
    gate = gate * up
    ffn_out = matmul_vec(dequant_tensor(tensors[f'blk.{layer}.ffn_down.weight']), gate, transposed=True)

    return (hidden + ffn_out).astype(np.float32)

def main():
    model_path = sys.argv[1]
    token_id = int(sys.argv[2]) if len(sys.argv) > 2 else 9707

    print(f"Loading TMAC from {model_path}...")
    tensors = load_tmac_tensors(model_path)
    print(f"Loaded {len(tensors)} tensors")

    # Get embedding
    print(f"\n=== EMBEDDING (token {token_id}) ===")
    hidden = get_embedding(tensors, token_id)
    print(f"hidden[0:5]: {hidden[0:5]}")
    print(f"mean: {hidden.mean():.6f}, std: {hidden.std():.6f}")
    hidden.tofile('/tmp/py_emb.bin')

    # KV cache
    k_cache = [np.zeros(K_DIM, dtype=np.float32) for _ in range(MAX_SEQ_LEN)]
    v_cache = [np.zeros(K_DIM, dtype=np.float32) for _ in range(MAX_SEQ_LEN)]

    # Forward through layers
    for layer in range(NUM_LAYERS):
        hidden = forward_layer(hidden, layer, 0, tensors, k_cache, v_cache)
        print(f"\n=== LAYER {layer} ===")
        print(f"hidden[0:5]: {hidden[0:5]}")
        print(f"mean: {hidden.mean():.6f}, std: {hidden.std():.6f}")
        hidden.tofile(f'/tmp/py_layer_{layer}.bin')
        print(f"  [DEBUG] hidden dtype={hidden.dtype} saved to /tmp/py_layer_{layer}.bin")

    # Apply output norm before lm_head
    output_norm = dequant_tensor(tensors['output_norm.weight'])
    norm_hidden = rms_norm(hidden, output_norm)
    norm_hidden.tofile('/tmp/py_output_norm.bin')
    print(f"\n=== OUTPUT NORM ===")
    print(f"norm_hidden[0:5]: {norm_hidden[0:5]}")
    print(f"mean: {norm_hidden.mean():.6f}, std: {norm_hidden.std():.6f}")
    print(f"Saved to /tmp/py_output_norm.bin")

    # Compute logits (lm_head with tied embeddings)
    emb = dequant_tensor(tensors['token_embd.weight'])
    logits = np.dot(norm_hidden, emb)  # [896] @ [896, 151936] = [151936]
    topk = np.argpartition(logits, -10)[-10:]
    topk = topk[np.argsort(-logits[topk])]
    print(f"\n=== TOP-10 LOGITS ===")
    for tid in topk:
        print(f"  {tid}: {logits[tid]:.3f}")

    print("\n=== DONE ===")
    print("Layer outputs saved to /tmp/py_layer_*.bin")
    print("Output norm saved to /tmp/py_output_norm.bin")

if __name__ == "__main__":
    main()