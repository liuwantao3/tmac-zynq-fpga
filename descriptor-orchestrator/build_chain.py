#!/usr/bin/env python3
"""
Multi-layer Chain Builder

Generates a full descriptor chain for all 28 layers.

Usage:
    python3 build_chain.py
    python3 build_chain.py -o /tmp/full_chain.bin
"""

import argparse
import struct
import sys
import json

DIMS = {
    "hidden_dim": 896,
    "intermediate_dim": 4864,
}

DESC_TYPE = {
    "INT16": 1,
    "Q8_0": 8,
    "Q5_0": 6,
    "Q6_K": 14,
    "Q4_K": 12,
    "CPU_OP": 15,
}

OP_TILE_ROWS = {
    "Q8_0": 64,
    "Q5_0": 8,
    "Q6_K": 32,
    "Q4_K": 56,
    "INT16": 64,
    "CPU_OP": 0,
}


def build_layer_config(layer_idx: int, buffer_base: int = 0x10000) -> dict:
    """Build config for one layer."""
    is_even = (layer_idx % 2 == 0)
    ffdn_qt = "Q6_K" if is_even else "Q4_K"
    ffdn_rows = OP_TILE_ROWS[ffdn_qt]
    
    bufs = {
        "hidden": {"offset": buffer_base, "size": 3584},
    }
    next_buf = buffer_base + 0x4000
    
    # Build 12 descriptors per layer
    ops = [
        {"name": "rms_norm", "weight": f"blk.{layer_idx}.attn_norm.weight", "in": "hidden", "out": "norm_out"},
        {"name": "matmul_q", "weight": f"blk.{layer_idx}.attn_q.weight", "bias": f"blk.{layer_idx}.attn_q.bias", "in": "norm_out", "out": "q"},
        {"name": "matmul_k", "weight": f"blk.{layer_idx}.attn_k.weight", "bias": f"blk.{layer_idx}.attn_k.bias", "in": "norm_out", "out": "k"},
        {"name": "matmul_v", "weight": f"blk.{layer_idx}.attn_v.weight", "bias": f"blk.{layer_idx}.attn_v.bias", "in": "norm_out", "out": "v"},
        {"name": "softmax_attention", "in": "q", "out": "context"},
        {"name": "matmul_output", "weight": f"blk.{layer_idx}.attn_output.weight", "in": "context", "out": "attn_out"},
        {"name": "residual_add", "in": "hidden", "out": "ffn_norm_out"},
        {"name": "matmul_gate", "weight": f"blk.{layer_idx}.ffn_gate.weight", "in": "ffn_norm_out", "out": "gate"},
        {"name": "matmul_up", "weight": f"blk.{layer_idx}.ffn_up.weight", "in": "ffn_norm_out", "out": "up"},
        {"name": "swiglu", "in": "gate", "out": "swiglu_out"},
        {"name": "matmul_down", "weight": f"blk.{layer_idx}.ffn_down.weight", "in": "swiglu_out", "out": "ffn_out"},
        {"name": "residual_add", "in": "hidden", "out": "hidden"},
    ]
    
    return {"layer": layer_idx, "ops": ops, "buffers": bufs}


def resolve_quant(op: dict, layer_idx: int) -> tuple:
    """Resolve operation to (quant_type, tile_rows)."""
    name = op["name"]
    cpu_ops = {"rms_norm", "rope", "softmax_attention", "residual_add", "swiglu"}
    
    if name in cpu_ops:
        return ("CPU_OP", 0)
    
    explicit = op.get("quant_type")
    if explicit:
        qt = explicit.upper()
    else:
        qt_map = {
            "matmul_q": "Q5_0",
            "matmul_k": "Q5_0",
            "matmul_v": "Q8_0",
            "matmul_output": "Q5_0",
            "matmul_gate": "Q5_0",
            "matmul_up": "Q5_0",
        }
        qt = qt_map.get(name, "INT16")
    
    if name == "matmul_down":
        qt = "Q6_K" if layer_idx % 2 == 0 else "Q4_K"
    
    return (qt, OP_TILE_ROWS.get(qt, 64))


def compile_layer(cfg: dict, layer_idx: int, dump: bool = False) -> bytes:
    """Compile one layer to binary."""
    ops = cfg["ops"]
    bufs = cfg.get("buffers", {})
    
    buf_offsets = {}
    for name, info in bufs.items():
        buf_offsets[name] = {"offset": info["offset"], "size": info["size"]}
    
    if "hidden" not in buf_offsets:
        buf_offsets["hidden"] = {"offset": 0x1000, "size": DIMS["hidden_dim"] * 4}
    
    descriptors = []
    
    for i, op in enumerate(ops):
        name = op["name"]
        inputs = op.get("in", "hidden")
        output = op.get("out", "hidden")
        
        if inputs not in buf_offsets:
            buf_offsets[inputs] = {"offset": 0x10000 + i * 0x1000, "size": DIMS["hidden_dim"] * 4}
        if output not in buf_offsets:
            buf_offsets[output] = {"offset": 0x10000 + i * 0x1000, "size": DIMS["hidden_dim"] * 4}
        
        in_buf = buf_offsets[inputs]
        out_buf = buf_offsets[output]
        
        qt, tile_rows = resolve_quant(op, layer_idx)
        
        weight_addr = 0
        if name.startswith("matmul"):
            wtensor = op.get("weight")
            if wtensor:
                weight_addr = 0x2000 + i * 0x2000
        
        result_addr = out_buf["offset"]
        quant_type_val = DESC_TYPE.get(qt, 1)
        flags = 0x02 if i == len(ops) - 1 else 0
        next_addr = 0xFFFFFFFF if i < len(ops) - 1 else 0
        act_bytes = DIMS["hidden_dim"] * 4
        
        if dump:
            q_str = f"{qt}"
            print(f"L{layer_idx:2d} D{i:2d}: {name:20s} {q_str:6s} rows={tile_rows:2d}")
            continue
        
        desc = struct.pack("<IIIIHHBBBHB6s",
                next_addr,
                weight_addr,
                in_buf["offset"],
                result_addr,
                1,
                0x1000,
                quant_type_val,
                tile_rows,
                flags,
                act_bytes,
                1,
                b'\x00' * 6
        )
        descriptors.append(desc)
    
    return b"".join(descriptors) if descriptors else None


def main():
    parser = argparse.ArgumentParser(description="Multi-layer chain builder")
    parser.add_argument("-o", "--output", help="Output binary file")
    parser.add_argument("-d", "--dump", action="store_true", help="Dump summary only")
    parser.add_argument("-n", "--layers", type=int, default=28, help="Number of layers")
    args = parser.parse_args()
    
    num_layers = 28  # Qwen2-0.5B has 28 layers
    
    print(f"[BUILD] Building {num_layers}-layer chain...")
    
    all_descriptors = []
    
    for layer_idx in range(num_layers):
        cfg = build_layer_config(layer_idx)
        layer_bin = compile_layer(cfg, layer_idx, args.dump)
        if layer_bin:
            all_descriptors.append(layer_bin)
        
        q_str = "Q6_K" if layer_idx % 2 == 0 else "Q4_K"
        print(f"  Layer {layer_idx:2d}: ffn_down={q_str}")
    
    full_chain = b"".join(all_descriptors)
    total_descs = num_layers * 12
    total_bytes = len(full_chain)
    
    print(f"[DONE] {total_bytes} bytes ({total_descs} descriptors)")
    
    if args.output:
        with open(args.output, "wb") as f:
            f.write(full_chain)
        print(f"Wrote to {args.output}")


if __name__ == "__main__":
    main()