# ARM Firmware (Aspirational)

ARM-side runtime for FPGA-accelerated LLM inference on Zynq 7010.

## Status
**Design phase** — not yet cross-compiled or deployed. The software-only C++ reference
implementation is at `sim/tmac_gguf.cpp`.

## Overview
The T-MAC runtime executes on the ARM Cortex-A9 processor and handles:
- Tokenization and embedding lookup
- LayerNorm and RMSNorm
- Softmax with attention
- Activation functions (SiLU)
- KV cache management
- Sampling and decoding
- FPGA offloading for GEMM operations (aspirational)

## Components
- `tmac_app.cpp` — Main entry point
- `tmac_runtime.cpp` / `tmac_runtime.hpp` — Core inference runtime (transformer forward pass)
- `tmac_fpga.cpp` / `tmac_fpga.hpp` — FPGA interface for matrix multiplication offloading

## Data Flow (Target)
```
DDR  →  ARM (dequant)  →  FPGA  →  ARM (residual/norm/act)  →  Result
```

## Memory Layout (Target, 512MB DDR)
```
0x00000000 - 0x00FFFFFF  (16MB):  ARM code + stack
0x01000000 - 0x01FFFFFF  (16MB):  Activation buffers
0x02000000 - 0x07FFFFFF  (96MB):  KV Cache
0x08000000 - 0x13FFFFFF (192MB):  Model weights
0x14000000 - 0x1FFFFFFF (192MB):  Reserved
```

## Building (Target)
```bash
cd firmware
arm-linux-gnueabihf-g++ -O3 -march=armv7-a -mfpu=neon \
    -o tmac_app tmac_app.cpp tmac_runtime.cpp tmac_fpga.cpp \
    -lpthread -lrt
```

## Reference
The verified C++ inference logic is in `sim/tmac_gguf.cpp`. The firmware mirrors
the same architecture but with FPGA offloading for GEMM operations.
