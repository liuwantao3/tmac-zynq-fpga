#!/usr/bin/env python3
"""
Vectorized Python inference using GGUF reader - generates layer-by-layer ground truth.
Uses numpy vectorization for speed.
"""

import sys
import numpy as np
from gguf import GGUFReader
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

def f16_to_f32_vec(arr):
    """Vectorized float16 to float32"""
    arr = arr.astype(np.uint16)
    sign = (arr >> 15) & 1
    exp = (arr >> 10) & 0x1F
    mant = arr & 0x3FF

    # Subnormal
    result = np.where(exp == 0,
                     (mant / 1024.0) * (2 ** -14),
                     (1 + mant / 1024.0) * (2 ** (exp - 15)))
    result = np.where(sign == 1, -result, result)
    result = np.where(exp == 31, np.inf, result)
    return result

def dequant_q8_0_block(data_bytes):
    """Dequantize Q8_0 block - data_bytes is raw bytes from GGUF"""
    n_elements = len(data_bytes) // 34 * 32

    # Parse scales (float16 at start of each 34-byte block)
    n_blocks = len(data_bytes) // 34
    scales = np.zeros(n_blocks, dtype=np.float32)
    for i in range(n_blocks):
        scale_raw = int(data_bytes[i*34]) | (int(data_bytes[i*34+1]) << 8)
        scales[i] = f16_to_f32_vec(np.array([scale_raw], dtype=np.uint16))[0]

    # Extract q8 values (bytes 2-33 of each block)
    q8_data = np.zeros(n_elements, dtype=np.int8)
    for i in range(n_blocks):
        block_q8 = np.frombuffer(bytes(data_bytes[i*34+2:i*34+34]), dtype=np.int8)
        start = i * 32
        end = min(start + 32, n_elements)
        q8_data[start:end] = block_q8[:end-start]

    # Broadcast scales to match q8 elements
    scales_expanded = np.repeat(scales, 32)[:n_elements]

    return scales_expanded * q8_data.astype(np.float32)

def load_tensor_raw(reader, name):
    """Load tensor raw data from GGUF"""
    for t in reader.tensors:
        if t.name == name:
            return reader.data[t.data_offset:t.data_offset + t.n_bytes]
    return None

def load_f32(reader, name):
    """Load float32 tensor"""
    raw = load_tensor_raw(reader, name)
    return np.frombuffer(raw.tobytes(), dtype=np.float32)

def main():
    model_path = sys.argv[1]
    token_id = int(sys.argv[2]) if len(sys.argv) > 2 else 9707

    print(f"Loading {model_path}...")
    reader = GGUFReader(model_path)

    # Load embedding
    print("Loading embedding...")
    emb_raw = None
    for t in reader.tensors:
        if t.name == 'token_embd.weight':
            emb_raw = reader.data[t.data_offset:t.data_offset + t.n_bytes]
            break

    # Token 9707 starts at byte offset 9707 * 952
    bytes_per_token = len(emb_raw) // 151936
    token_bytes = emb_raw[token_id * bytes_per_token:(token_id + 1) * bytes_per_token]

    # Dequantize embedding for token
    emb = dequant_q8_0_block(bytes(token_bytes))
    print(f"\n=== EMBEDDING ===")
    print(f"shape: {emb.shape}")
    print(f"emb[0:5]: {emb[0:5]}")
    print(f"mean: {emb.mean():.6f}, std: {emb.std():.6f}")

    emb.tofile('/tmp/py_emb.bin')
    print("Saved embedding to /tmp/py_emb.bin")

    # Load attn_norm weights
    print("\nLoading attn_norm weights...")
    attn_norms = []
    for layer in range(NUM_LAYERS):
        w = load_f32(reader, f'blk.{layer}.attn_norm.weight')
        attn_norms.append(w)

    # RMS norm
    def rms_norm(x, weight):
        eps = 1e-5
        rms = np.sqrt(np.mean(x**2) + eps)
        return x * weight / rms

    # Process through layers
    hidden = emb.copy()
    k_cache = [np.zeros(K_DIM) for _ in range(256)]
    v_cache = [np.zeros(K_DIM) for _ in range(256)]

    print(f"\n=== LAYER 0 INPUT ===")
    print(f"hidden[0:5]: {hidden[0:5]}")
    print(f"mean: {hidden.mean():.6f}, std: {hidden.std():.6f}")

    for layer in range(NUM_LAYERS):
        original = hidden.copy()

        # Attention norm
        hidden = rms_norm(hidden, attn_norms[layer])

        # For now, just record the norm output
        print(f"\n=== LAYER {layer} ===")
        print(f"After attn_norm: mean={hidden.mean():.6f}, std={hidden.std():.6f}")

        # Save layer output
        hidden.tofile(f'/tmp/py_layer_{layer}.bin')

        # For a full forward pass, we need Q,K,V projections etc.
        # But let's first verify just the embedding and norm are correct

    print("\n=== SUMMARY ===")
    print("Saved layer outputs to /tmp/py_layer_*.bin")
    print("Run TMAC and compare at each layer")

if __name__ == "__main__":
    main()