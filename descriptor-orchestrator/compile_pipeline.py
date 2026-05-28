#!/usr/bin/env python3
"""
Full Pipeline Compiler

Compiles the full pipeline (embedding + layers + logits) to binary.

Usage:
    python3 compile_pipeline.py examples/model.json -o /tmp/pipeline.bin
    python3 compile_pipeline.py examples/model.json --dump
"""

import argparse
import json
import struct
import sys

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

# Load FPGA capabilities from JSON if available
def load_fpga_caps(path: str = "../fpga_caps.json") -> dict | None:
    """Load FPGA capabilities from JSON config."""
    import os
    full_path = os.path.join(os.path.dirname(__file__), path)
    if os.path.exists(full_path):
        with open(full_path) as f:
            return json.load(f)
    return None

def validate_fpga_caps(caps: dict, tile_specs: dict) -> bool:
    """Validate that tile sizes in fpga_caps.json match compiler defaults."""
    errors = []
    for qt, rows in tile_specs.items():
        if qt in caps.get("tile_sizes", {}):
            cap_rows = caps["tile_sizes"][qt].get("rows")
            if cap_rows != rows:
                errors.append(f"{qt}: JSON={cap_rows}, compiler={rows}")
    if errors:
        print("[WARNING] FPGA caps mismatch:")
        for e in errors:
            print(f"  - {e}")
        return False
    else:
        print("[CAPS] Tile sizes validated OK")
    return True

DESC_TYPE["token_embedding"] = "Q8_0"


def resolve_quant(op: dict, base_path: str = ".") -> tuple:
    """Resolve operation to (quant_type, tile_rows)."""
    name = op["name"]
    
    # CPU-only operations
    if name in ["rms_norm", "rope", "softmax_attention", "residual_add", "swiglu"]:
        return ("CPU_OP", 0)
    
    # Check explicit quant_type first
    if "quant_type" in op:
        qt = op["quant_type"].upper()
        return (qt, OP_TILE_ROWS.get(qt, 64))
    
    # Determine from op name patterns
    if name == "matmul_q" or name == "matmul_output":
        return ("Q5_0", 8)
    elif name == "matmul_k":
        return ("Q5_0", 8)  
    elif name == "matmul_v":
        return ("Q8_0", 64)
    elif name == "matmul_gate" or name == "matmul_up":
        return ("Q5_0", 8)
    elif name == "matmul_down":
        # Determine from layer file name: layer_XX.json
        layer_num = int(base_path.split("_")[-1].split(".")[0]) if "_" in base_path else 0
        qt = "Q6_K" if layer_num % 2 == 0 else "Q4_K"
        tr = 32 if layer_num % 2 == 0 else 56
        return (qt, tr)
    elif "token" in name or "lm_head" in name:
        return ("Q8_0", 64)
    
    return ("INT16", 64)


def compile_stage(stage_cfg: dict, base_path: str, dump: bool, offset: int) -> bytes:
    """Compile one stage to binary."""
    ops = stage_cfg.get("ops", [])
    if not ops:
        return b"", offset
    
    descriptors = []
    buf_offset = 0x1000
    
    for i, op in enumerate(ops):
        name = op["name"]
        inputs = op.get("in", "hidden")
        output = op.get("out", "hidden")
        
        qt, tile_rows = resolve_quant(op, base_path)
        
        weight_addr = 0
        if "weight" in op:
            weight_addr = 0x2000 + i * 0x2000
        
        result_addr = buf_offset
        quant_type_val = DESC_TYPE.get(qt, 1)
        flags = 0x02 if i == len(ops) - 1 else 0
        next_addr = 0xFFFFFFFF if i < len(ops) - 1 else 0
        act_bytes = 896 * 4
        
        if dump:
            print(f"  {name:20s} {qt:6s} rows={tile_rows}")
            continue
        
        desc = struct.pack("<IIIIHHBBBHB6s",
                next_addr,
                weight_addr,
                buf_offset,
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
        buf_offset += 0x1000
    
    return b"".join(descriptors), offset + len(descriptors) * 32


def compile_layer_ref(ref_path: str, base_path: str, dump: bool, offset: int) -> tuple:
    """Compile a layer reference."""
    layer_path = f"{base_path}/{ref_path}"
    with open(layer_path) as f:
        layer = json.load(f)
    return compile_stage(layer, base_path, dump, offset)


def compile_pipeline(config_path: str, dump_only: bool = False) -> bytes:
    """Compile full pipeline."""
    import os
    base_path = os.path.dirname(config_path)
    if not base_path:
        base_path = "."
    
    with open(config_path) as f:
        cfg = json.load(f)
    
    print(f'[PIPELINE] {cfg["model"]}')
    print(f'Stages: {len(cfg["stages"])}')
    
    all_binary = []
    desc_offset = 0
    
    for stage in cfg["stages"]:
        name = stage["name"]
        
        if "ops" in stage:
            print(f'Compiling stage: {name} ({len(stage["ops"])} ops)')
            binary, desc_offset = compile_stage(stage, base_path, dump_only, desc_offset)
            if not dump_only:
                all_binary.append(binary)
        elif "ref" in stage:
            print(f'Compiling layer: {name} ref={stage["ref"]}')
            binary, desc_offset = compile_layer_ref(stage["ref"], base_path, dump_only, desc_offset)
            if not dump_only:
                all_binary.append(binary)
    
    if dump_only:
        print(f'\nTotal descriptors: {desc_offset // 32}')
        return None
    
    return b"".join(all_binary)


def main():
    parser = argparse.ArgumentParser(description="Full pipeline compiler")
    parser.add_argument("config", help="Pipeline config (model.json)")
    parser.add_argument("-o", "--output", help="Output binary")
    parser.add_argument("-d", "--dump", action="store_true", help="Dump summary")
    parser.add_argument("--caps", help="FPGA capabilities JSON", default="../fpga_caps.json")
    args = parser.parse_args()
    
    # Load and validate FPGA caps
    caps = load_fpga_caps(args.caps)
    if caps:
        print(f'[CAPS] Loaded FPGA capabilities from {args.caps}')
        validate_fpga_caps(caps, OP_TILE_ROWS)
    else:
        print('[CAPS] Using built-in defaults (no fpga_caps.json found)')
    
    binary = compile_pipeline(args.config, args.dump)
    
    if args.dump:
        return
    
    if args.output:
        with open(args.output, "wb") as f:
            f.write(binary)
        print(f'\nWrote {len(binary)} bytes to {args.output}')
    else:
        sys.stdout.buffer.write(binary)


if __name__ == "__main__":
    main()