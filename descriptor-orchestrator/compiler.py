#!/usr/bin/env python3
"""
Layer Descriptor Chain Compiler

Reads a layer config (YAML/JSON) and generates a descriptor chain binary
suitable for loading into FPGA DDR.

Usage:
    python3 compiler.py layer.yaml -o descriptors.bin
    python3 compiler.py layer.json --dump      # just print summary
    python3 compiler.py layer.json -o out.bin  # write binary
"""

import argparse
import json
import struct
import sys

DIMS = {
    "hidden_dim": 896,
    "intermediate_dim": 4864,
    "num_heads": 14,
    "num_kv_heads": 2,
    "head_dim": 64,
}

DESC_TYPE = {
    "INT16": 1,
    "Q8_0": 8,
    "Q5_0": 6,
    "Q6_K": 14,
    "Q4_K": 12,
    "CPU_OP": 15,  # CPU-only operation: FPGA signals CPU and waits
}

CPU_OPS = ["rms_norm", "rope", "softmax_attention", "residual_add", "swiglu"]

OP_TILE_ROWS = {
    "Q8_0": 64,
    "Q5_0": 8,
    "Q6_K": 32,
    "Q4_K": 56,
    "INT16": 64,
}


def parse_config(path: str) -> dict:
    """Load and parse config file."""
    with open(path) as f:
        if path.endswith(".yaml") or path.endswith(".yml"):
            try:
                import yaml
                return yaml.safe_load(f)
            except ImportError:
                print("ERROR: PyYAML not installed. Install: pip install pyyaml")
                sys.exit(1)
        else:
            return json.load(f)


def resolve_quant_type(op_name: str, layer_idx: int, explicit: str = None) -> tuple:
    """Map operation to (quant_type_name, tile_rows)."""
    # CPU-only operations are handled by CPU, not FPGA
    cpu_ops = ["rms_norm", "rope", "softmax_attention", "residual_add", "swiglu"]
    if op_name in cpu_ops:
        return ("CPU_OP", 0)
    
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
            "add_bias": "INT16",
        }
        qt = qt_map.get(op_name, "INT16")
    
    if op_name == "matmul_down":
        qt = "Q6_K" if layer_idx % 2 == 0 else "Q4_K"
    
    tile_rows = OP_TILE_ROWS.get(qt, 64)
    return (qt, tile_rows)


def compile_descriptors(cfg: dict, dump_only: bool = False) -> bytes | None:
    """Compile config to descriptor chain binary."""
    layer = cfg["layer"]
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

        qt, tile_rows = resolve_quant_type(name, layer, op.get("quant_type"))
        act_bytes = DIMS["hidden_dim"] * 4

        weight_addr = 0
        if name.startswith("matmul"):
            wtensor = op.get("weight")
            if wtensor:
                weight_addr = 0x2000 + i * 0x2000

        result_addr = out_buf["offset"]
        quant_type_val = DESC_TYPE.get(qt, 1)
        flags = 0x02 if i == len(ops) - 1 else 0
        next_addr = 0xFFFFFFFF if i < len(ops) - 1 else 0

        # Print summary in dump mode
        if dump_only:
            print(f"Desc {i:2d}: {name:20s} in={inputs} out={output} "
                  f"type={quant_type_val}({qt}) rows={tile_rows} "
                  f"in_off={in_buf['offset']:#x} res_off={result_addr:#x}")
            continue

        # PhaseBDescriptor (32 bytes): 4I + 2H + 3B + H + B + 6s
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
    parser = argparse.ArgumentParser(description="Layer descriptor chain compiler")
    parser.add_argument("config", help="Config file (YAML or JSON)")
    parser.add_argument("-o", "--output", help="Output binary file")
    parser.add_argument("-d", "--dump", action="store_true", help="Dump descriptor summary only")
    args = parser.parse_args()

    cfg = parse_config(args.config)
    data = compile_descriptors(cfg, args.dump)

    if args.dump:
        print(f"[SUMMARY] {len(cfg['ops'])} descriptors for layer {cfg['layer']}")
        return

    if args.output:
        with open(args.output, "wb") as f:
            f.write(data)
        print(f"Wrote {len(data)} bytes ({len(cfg['ops'])} descriptors) to {args.output}")
    else:
        sys.stdout.buffer.write(data)


if __name__ == "__main__":
    main()