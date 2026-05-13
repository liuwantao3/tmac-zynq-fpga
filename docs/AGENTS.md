# FPGA Design Skill

## Overview

This skill automates FPGA development workflows using Xilinx Vivado Design Suite and Vitis HLS with a three-layer feedback system.

## Hardware Target

- **Device**: Xilinx Zynq 7010 (xc7z010clg400-1)
- **Constraints**:
  - DSP slices: 80 (64 used for 8×8 systolic array, 16 spare)
  - BRAM: 240 KB (60 × BRAM 18K blocks)
  - LUTs: 17,600
  - FF: 35,200
  - DDR3: 512 MB
- **Application**: Qwen 0.5B LLM inference on edge

## Supported Configurations

| Config Register | Values | Description |
|----------------|--------|-------------|
| `op_vecmul` | 0=MatMul, 1=VecMul | Operation mode |

### Use Cases
- **Prefill**: Matrix-matrix (batch processing)
- **Decode**: Vector-matrix (single token generation)

## HLS Kernel: matmul_q8 (Primary)

N=64 fixed, 8×8 systolic sub-blocking, LUT-based scale multipliers.

```cpp
void matmul_q8(
    ap_int<8> A[N * N],                         // Q8_0 weight bytes (4 KB)
    combined_scale_t combined_scales[N * 2],    // 128 × UQ8.8 fixed-point (256 B)
    in_t X[N],                                  // Activation vector (INT16, 128 B)
    acc_t Y[N],                                 // Output vector (INT64, 512 B)
    volatile ap_uint<32> *control,              // Control register
    volatile ap_uint<32> *status,               // Status register
    ap_uint<1> &interrupt                       // Interrupt output
);
```

### Control Register (32-bit AXI slave)
| Bit | Name | Description |
|-----|------|-------------|
| [0] | start | Write 1 to start operation |
| [1] | int_enable | Enable interrupt on completion |
| [2] | reserved | — |
| [3] | op_vecmul | 0=MatMul, 1=VecMul |
| [31:4] | reserved | Reserved |

### Status Register (32-bit AXI slave)
| Value | Name | Description |
|-------|------|-------------|
| 0 | STATUS_IDLE | Idle, ready for command |
| 1 | STATUS_RUNNING | Computation in progress |
| 2 | STATUS_DONE | Computation completed |
| 3 | STATUS_ERROR | Error occurred |

### Communication Protocol
1. **CPU**: Write A (Q8_0 weights), combined_scales, X (activation) to AXI master ports
2. **CPU**: Write control register (start=1, int_enable=1)
3. **FPGA**: Set status=STATUS_RUNNING
4. **FPGA**: Q8→INT16 dequant (LUT) + INT16 systolic matmul
5. **FPGA**: Set status=STATUS_DONE, trigger interrupt if enabled
6. **CPU**: Read Y (result) from AXI master port

### Interrupt Behavior
- Long operations (>10μs): Interrupt driven (CPU free during computation)
- Short operations: Poll status register (avoid interrupt overhead)

### Data Types (Q8_0 kernel)
- `in_t`: ap_int<16> (INT16 activation / dequantized weight)
- `prod_t`: ap_int<32> (INT16×INT16 product)
- `acc_t`: ap_int<64> (INT64 accumulator — avoids overflow in 64-deep dot product)

### Architecture
- 8×8 systolic array (64 DSP)
- Blocked matrix multiplication (8×8 sub-blocks)
- BRAM for weight and activation buffers (Zynq 7010 has no URAM)

## Software FPGA Simulation (`sim/tmac_gguf.cpp`)

A standalone C++17 simulation of the full inference pipeline with FPGA accelerator modeling.

### Build
```bash
cd sim && g++ -std=c++17 -pthread -O2 -o tmac_gguf tmac_gguf.cpp matmul_q8.cpp
```

### Usage
```bash
echo 9707 | ./tmac_gguf /path/to/model.tmac              # FP32 (ground truth)
echo 9707 | ./tmac_gguf /path/to/model.tmac --fpga        # INT8 FPGA sim
echo 9707 | ./tmac_gguf /path/to/model.tmac --fpga-int16  # INT16 FPGA sim (recommended)
echo 9707 | ./tmac_gguf /path/to/model.tmac --generate 10 # autoregressive generation
echo 9707 | ./tmac_gguf /path/to/model.tmac --dump-layers # layer-by-layer dump
```

### Flags
| Flag | Description |
|------|-------------|
| `--fpga` | INT8 quantized matmul via simulated systolic array |
| `--fpga-int16` | INT16 quantized matmul (highest FPGA accuracy) |
| `--fpga-q8` | Q8_0 direct path — FPGA receives raw Q8_0 bytes, does LUT-based Q8→INT16 dequant + INT16 systolic matmul |
| `--generate N` | Generate N tokens autoregressively |
| `--perf` | Enable pipeline profiling (Chrome trace JSON + bottleneck analysis) |
| `--dump-layers` | Save hidden state after each layer to `/tmp/` |

### INT8 vs INT16 Accuracy (Qwen2-0.5B, single token)
| Metric | INT8 | INT16 |
|--------|------|-------|
| Max logit diff vs FP32 | 35.6 | **0.24** |
| Mean logit diff | 12.4 | **0.035** |
| Top-5 token match | 0/5 | **5/5** |
| Top-1 correct | No | **Yes** |

**Why INT8 fails**: SwiGLU in the FFN amplifies quantization error multiplicatively (silu(gate)×up), causing catastrophic divergence by layer 3. INT8's 256 levels are insufficient.

**Why INT16 works**: 65536 levels (256× finer) keep SwiGLU error bounded. Zynq 7010 DSP48E1 handles INT16×INT16 in one slice.

### Timing (single token, Apple M2)
| Mode | CPU Time | Est. FPGA Time (150 MHz) | FPGA Speedup |
|------|----------|-------------------------|--------------|
| FP32 naive | 11.0s | — | — |
| INT16 sim | 19.7s | 51.5 ms | ~214× vs CPU |
| Q8_0 sim | 23.5s | 482.8 ms | ~49× vs CPU |

**Note**: Q8_0 path is slower in simulation because ARM-side does full dequant to compute row_max_abs + combined scales. On real FPGA, ARM only does memcpy (no math), and the 82 ms FPGA time covers all 906,836 tiles × 214 cycles/tile @ 150 MHz.

## LLM Inference Application

### Memory Analysis
```
Parameters: 0.5B parameters, 373.7 MB (TMAC format, Q5_0/Q6_K mixed)
Activations + KV cache: ~150-200 MB
Total: ~400-450 MB (within 512 MB DDR)
```

### Performance Targets
| Mode | Latency | Throughput |
|------|---------|------------|
| Prefill | ~48ms/token | ~20 tokens/s |
| Decode | ~35ms/token | ~28 tokens/s |

### Resource Usage (Zynq 7010) — Q8_0 LUT Path
| Resource | Used | Available | Usage |
|----------|------|----------|-------|
| DSP | 64 | 80 | 80% |
| LUT | ~14K | 17,600 | ~80% |
| BRAM | ~36 KB | 135 KB | ~27% |
| FF | ~16K | 35,200 | ~45% |

## Directory Structure

```
/Users/arctic/fpga/
├── hls/
│   ├── matmul_q8.cpp        # Primary HLS kernel (Q8_0 direct path, LUT scale mult)
│   ├── matmul_q8.hpp        # Q8 kernel type/constant defines
│   ├── matmul_int16.cpp     # INT16 kernel (legacy fallback)
│   ├── matmul_int8.cpp      # INT8 kernel (deprecated)
│   ├── script_q8.tcl        # Q8 kernel synthesis script (recommended)
│   ├── script_int16.tcl     # INT16 kernel synthesis script
│   ├── script.tcl           # INT8 kernel synthesis script (deprecated)
│   ├── hls_config.tcl       # Shared HLS directives (config_bind -mul_style luts)
├── vivado/
│   ├── block_design.tcl     # Vivado block design
│   └── hls_ip.tcl           # HLS IP integration
├── scripts/
│   ├── design_iteration.sh   # Iteration loop wrapper
│   └── feedback_parser.py    # Three-layer feedback parser
├── Makefile
└── docs/
    └── AGENTS.md             # This file
```

## Three-Layer Feedback System

### Layer 1: Console Output
Real-time stdout/stderr from Vivado/Vitis HLS compilation

### Layer 2: Report Files
- HLS: `solution/solution.xml`, `solution/solution_report.xml`
- Vivado: `design_2_utilization_routed.rpt`, `design_2_timing_summary.rpt`

### Layer 3: Tcl Query Interface
Direct queries to Vivado for design exploration

## Commands

### HLS Synthesis
```bash
cd /Users/arctic/fpga && ./scripts/design_iteration.sh hls
# or: make hls
```

### Vivado Implementation
```bash
cd /Users/arctic/fpga && ./scripts/design_iteration.sh vivado
# or: make vivado
```

### Full Pipeline
```bash
cd /Users/arctic/fpga && ./scripts/design_iteration.sh all
# or: make all
```

### Get Structured Feedback
```bash
python3 /Users/arctic/fpga/scripts/feedback_parser.py
```

### Iteration Loop
```bash
make iterate
```

## Feedback Parser Output (JSON)

```json
{
  "timestamp": "2026-05-05 10:30:00",
  "hls": {
    "latency_cycles": 1234,
    "DSP_used": 45,
    "DSP_percent": 20.5,
    "LUT_used": 1234,
    "LUT_percent": 4.4,
    "BRAM_used": 5,
    "BRAM_percent": 8.3
  },
  "vivado_util": {
    "utilized": {"LUT": 5000, "DSP": 45, "BRAM": 5, "FF": 8000},
    "percent": {"LUT": 17.9, "DSP": 20.5, "BRAM": 8.3, "FF": 22.7}
  },
  "vivado_timing": {
    "wns": 2.5,
    "tns": 0.0,
    "ws": 1.2
  },
  "recommendations": [
    {
      "type": "DSP",
      "severity": "warning",
      "issue": "DSP utilization at 85% (threshold: 80%)",
      "action": "Reduce unroll factor in HLS pragma"
    }
  ]
}
```

## Optimization Strategies

| Issue | Detection | Action |
|-------|-----------|--------|
| DSP > 80% | HLS report / feedback | Reduce unroll factor or systolic array size |
| LUT > 85% | Vivado utilization | Enable logic optimization, reduce vecmul UNROLL factor |
| Latency too high | HLS latency > 10000 | Add PIPELINE pragma |
| WNS < 0 | Vivado timing | Increase clock period or balance registers |
| BRAM > 70% | Vivado utilization | Memory partitioning |

## Typical HLS Pragmas (Q8_0 LUT Path)

```cpp
// Pipeline the inner loop
#pragma HLS PIPELINE II=1

// Unroll for parallel systolic execution
#pragma HLS UNROLL factor=8

// Force multipliers to LUT (preserve DSPs for systolic array)
// Set in TCL: config_bind -mul_style luts

// BRAM for weight storage (Zynq 7010 has no URAM)
#pragma HLS RESOURCE variable=A core=RAM_2P_BRAM

// Array partitioning
#pragma HLS ARRAY_PARTITION variable=data dim=1 factor=4
```

## Iteration Workflow

1. **Run**: Execute current design (HLS or Vivado)
2. **Capture**: Collect Layer 1 console output
3. **Parse**: Run `feedback_parser.py` for Layer 2 reports
4. **Analyze**: Review JSON output for metrics and recommendations
5. **Modify**: Update source code or pragmas
6. **Repeat**: Until constraints are met

## Quick Test

```bash
# Full pipeline with feedback
cd /Users/arctic/fpga && make iterate

# Clean and rebuild
cd /Users/arctic/fpga && make clean && make iterate
```

## Tool Paths

- Vitis HLS: `/opt/Xilinx/Vitis_HLS/2023.1/bin/vitis_hls`
- Vivado: `/opt/Xilinx/Vivado/2023.1/bin/vivado`