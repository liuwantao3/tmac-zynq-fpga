#!/usr/bin/env python3
"""
Compare dequantization: gguf library vs llama.cpp
Goal: Find why they differ and fix C++ dequantization
"""

import sys
import numpy as np
import struct

try:
    from gguf import GGUFReader, GGMLQuantizationType
    from gguf.quants import dequantize
except ImportError:
    print("[ERROR] pip install gguf")
    sys.exit(1)

def f16_to_f32(f16):
    """Convert F16 (uint16) to F32"""
    u16 = f16 & 0xFFFF
    sign = (u16 >> 15) & 0x1
    exp = (u16 >> 10) & 0x1F
    mant = u16 & 0x3FF
    if exp == 0:
        if mant == 0:
            return 0.0
        return (sign * -2 + 1) * (mant / 1024.0) * 2**(-14)
    elif exp == 31:
        return float('nan') if mant else (float('-inf') if sign else float('inf'))
    else:
        return (sign * -2 + 1) * (1 + mant / 1024.0) * 2**(exp - 15)


def dequant_q8_0_manual(data, idx):
    """Manual Q8_0 dequantization matching llama.cpp"""
    block_size = 34  # 2 bytes scale + 32 bytes q8
    block = idx // 32
    offset = idx % 32
    block_offset = block * block_size
    
    # Read scale (f16)
    scale_raw = data[block_offset] | (data[block_offset + 1] << 8)
    scale = f16_to_f32(scale_raw)
    
    # Read quantized value (int8)
    q8_val = data[block_offset + 2 + offset]
    if q8_val > 127:
        q8_val -= 256
    
    return scale * q8_val / 127.0


def dequant_q5_0_manual(data, idx):
    """Manual Q5_0 dequantization"""
    block_size = 22  # 2 bytes scale + 4 bytes qh + 16 bytes qs
    block = idx // 32
    offset = idx % 32
    base = block * block_size
    
    # Read scale
    scale_raw = data[base] | (data[base + 1] << 8)
    scale = f16_to_f32(scale_raw)
    
    # Read qh (4 bytes)
    qh = (data[base + 2] | (data[base + 3] << 8) | 
          (data[base + 4] << 16) | (data[base + 5] << 24))
    
    # Read quantized value (4 bits per value, stored in qs)
    qs_byte = data[base + 6 + offset // 2]
    ql = (qs_byte & 0xF) if offset % 2 == 0 else (qs_byte >> 4)
    
    # Read high bit from qh
    qh_bit = (qh >> offset) & 1
    
    # Combine: q = (qh_bit << 4) | ql, then subtract 16
    q = ((qh_bit << 4) | ql) - 16
    
    return scale * q


def dequant_q6_k_manual(data, idx):
    """Manual Q6_K dequantization"""
    block_size = 210  # Complex format
    QK_K = 256
    block = idx // QK_K
    offset = idx % QK_K
    base = block * block_size
    
    # Read scale (f16 at offset 208)
    scale_raw = data[base + 208] | (data[base + 209] << 8)
    d = f16_to_f32(scale_raw)
    
    # Read low 4 bits (stored in first 128 bytes, 2 values per byte)
    qs_byte = data[base + offset // 2]
    ql = (qs_byte & 0xF) if offset % 2 == 0 else (qs_byte >> 4)
    
    # Read high 2 bits (stored in bytes 128-191, 4 values per byte)
    qh_byte = data[base + 128 + offset // 4]
    qh_shift = (offset % 4) * 2
    qh = (qh_byte >> qh_shift) & 0x3
    
    # Combine: q = (qh << 4) | ql, then subtract 32
    q = ((qh << 4) | ql) - 32
    
    # Read scale for this element (stored in bytes 192-207, 1 byte per 16 elements)
    scale_idx = offset // 16
    scale_val = data[base + 192 + scale_idx]
    if scale_val > 127:
        scale_val -= 256
    
    return d * scale_val * q / 127.0


def compare_dequant():
    """Compare dequantization methods"""
    print("Loading GGUF file...")
    reader = GGUFReader('/Users/arctic/Downloads/qwen2-0_5b-instruct-q4_k_m.gguf')
    
    # Test token_embd.weight (Q8_0)
    for t in reader.tensors:
        if t.name == 'token_embd.weight':
            print(f"\nTesting: {t.name}")
            print(f"Type: {t.tensor_type} (Q8_0)")
            print(f"Shape: {t.shape}")
            
            # Get raw data
            raw_data = bytes(t.data)
            print(f"Raw data size: {len(raw_data)} bytes")
            
            # Dequantize with gguf library
            gguf_dequant = dequantize(t.data, GGMLQuantizationType(t.tensor_type))
            print(f"gguf library shape: {gguf_dequant.shape}")
            print(f"gguf library [0][:10]: {gguf_dequant[0][:10]}")
            
            # Manual dequantization for first 100 values
            manual_vals = []
            for i in range(100):
                val = dequant_q8_0_manual(raw_data, i)
                manual_vals.append(val)
            
            print(f"Manual [0:10]: {manual_vals[:10]}")
            
            # Compare
            print(f"\nComparison (first 10):")
            for i in range(10):
                g = gguf_dequant[0][i]
                m = manual_vals[i]
                print(f"  [{i}] gguf={g:.6f}, manual={m:.6f}, diff={abs(g-m):.6f}")
            
            # Check if they're the same
            max_diff = max(abs(gguf_dequant[0][i] - manual_vals[i]) for i in range(100))
            print(f"\nMax diff (first 100): {max_diff:.6f}")
            break
    
    # Test a Q5_0 tensor (e.g., attn_q.weight)
    print("\n" + "="*60)
    for t in reader.tensors:
        if 'attn_q' in t.name and 'weight' in t.name:
            print(f"\nTesting: {t.name}")
            print(f"Type: {t.tensor_type}")
            
            raw_data = bytes(t.data)
            gguf_dequant = dequantize(t.data, GGMLQuantizationType(t.tensor_type))
            print(f"gguf library shape: {gguf_dequant.shape}")
            print(f"gguf library [0][:10]: {gguf_dequant[0][:10]}")
            
            # Manual dequant for first 100 values
            manual_vals = []
            for i in range(min(100, gguf_dequant.size)):
                val = dequant_q5_0_manual(raw_data, i)
                manual_vals.append(val)
            
            print(f"Manual [0:10]: {manual_vals[:10]}")
            
            # Compare
            max_diff = max(abs(gguf_dequant.flat[i] - manual_vals[i]) for i in range(min(100, gguf_dequant.size)))
            print(f"Max diff (first 100): {max_diff:.6f}")
            break
    
    # Test a Q6_K tensor (e.g., ffn_down.weight)
    print("\n" + "="*60)
    for t in reader.tensors:
        if 'ffn_down' in t.name and 'weight' in t.name:
            print(f"\nTesting: {t.name}")
            print(f"Type: {t.tensor_type}")
            
            raw_data = bytes(t.data)
            gguf_dequant = dequantize(t.data, GGMLQuantizationType(t.tensor_type))
            print(f"gguf library shape: {gguf_dequant.shape}")
            print(f"gguf library [0][:10]: {gguf_dequant[0][:10]}")
            
            # Manual dequant for first 100 values
            manual_vals = []
            for i in range(min(100, gguf_dequant.size)):
                val = dequant_q6_k_manual(raw_data, i)
                manual_vals.append(val)
            
            print(f"Manual [0:10]: {manual_vals[:10]}")
            
            # Compare
            max_diff = max(abs(gguf_dequant.flat[i] - manual_vals[i]) for i in range(min(100, gguf_dequant.size)))
            print(f"Max diff (first 100): {max_diff:.6f}")
            break


if __name__ == "__main__":
    compare_dequant()
