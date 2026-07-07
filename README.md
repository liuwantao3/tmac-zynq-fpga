# Qwen2-0.5B FPGA Accelerator (Zynq 7010)

Multi-core Verilog RTL accelerator for Qwen2-0.5B-Instruct inference on the
MicroPhase Z7-Lite (xc7z010clg400-1). All 6 compute cores + HP descriptor-chain
DMA engine **synthesized and verified on hardware**: 18 HW tests pass, 123 simulation
tests pass.

| Component | Status |
|-----------|--------|
| Verilog RTL (6 cores + HP FSM) | Synthesized, bitstream built |
| HP FSM descriptor-chain DMA (AXI4-Lite + HP0) | 18 HW tests PASS |
| Q8_0 core (64x896, 6-stage pipeline) | 6/6 sim + HW verified |
| Q5_0 core (4x896, 2-core block-streaming) | 10/10 dispatch sim + HW verified |
| Q4_K core (56x256 block decode) | 4/4 sim |
| Q6_K core (32x256 block decode) | 97/97 sim |
| INT16 core (64x64 general) | Sim only (pre-existing wmem bug) |
| C++ inference engine (`sim/tmac_gguf`) | FP32 < 0.0003 vs ground truth |

## Architecture

```
CPU (ARM Cortex-A9)
    │
    ├── GP0 (AXI4-Lite @ 0x43C00000) → hp_fsm_top.v control/status registers
    │
    └── HP0 (AXI3, 32-bit physical) → hp_fsm_top.v DMA engine
                                        ├── Descriptor chain parser
                                        ├── Q8 compute dispatch (matmul_q8_core)
                                        ├── Q5_0 compute dispatch (2× matmul_q5_0_core)
                                        └── CPU_OP passthrough
```

### Quantization Types

| tensor | shape | type | Verilog core |
|--------|-------|------|-------------|
| `token_embd` | 151936x896 | Q8_0 | `matmul_q8_core.v` |
| `attn_v` | 128x896 | Q8_0 | `matmul_q8_core.v` |
| `attn_q`, `attn_k`, `attn_output` | 896x896 | Q5_0 | `matmul_q5_0_core.v` |
| `ffn_gate`, `ffn_up` | 4864x896 | Q5_0 | `matmul_q5_0_core.v` |
| `ffn_down` (even layers) | 896x4864 | Q6_K | `matmul_q6_k_core.v` |
| `ffn_down` (odd layers) | 896x4864 | Q4_K | `matmul_q4k_core.v` |

### Multi-Tile Descriptors

One descriptor can process multiple tiles (eliminates thousands of redundant DDR
reads). Set `num_tiles` at descriptor bytes [22:23]. One Q5_0 descriptor with
`num_tiles=224` processes the full 896x896 `attn_q` in a single descriptor.

## Quick Start

```bash
# Build C++ inference engine
g++ -std=c++17 -pthread -O2 -I sim -I gguf -I . \
    sim/tmac_gguf.cpp sim/matmul_q8.cpp -o sim/tmac_gguf

# Convert GGUF model
python3 scripts/extract_tmac.py models/qwen2-0_5b-instruct-q4_k_m.gguf /tmp/model.tmac

# Run inference (all FPGA simulation paths)
echo "9707" | ./sim/tmac_gguf /tmp/model.tmac --fpga-q8 --fpga-q5-0 --fpga-q6-k --fpga-q4k

# Run Verilog simulation tests
make -C verilog all
```

## Repository Structure

```
├── AGENTS.md                          ← Full architecture, register map, debug guide
├── vivado_integration/
│   ├── API.md                         ← Hardware API: registers, descriptor format, DDR layout
│   ├── build_bd.tcl                   ← Vivado batch build script
│   ├── rtl/hp_fsm_top.v              ← Active top: HP descriptor-chain FSM + Q8 + Q5_0
│   └── sw/
│       ├── run_hp_fsm_comprehensive.tcl  ← 10 HW tests (Q8 + DMA)
│       ├── run_hp_fsm_q5_0.tcl        ← 3 HW tests (Q5_0)
│       └── regs.h                     ← Register map constants
├── verilog/
│   ├── matmul_q8_core.v              ← Q8_0 6-stage pipeline core
│   ├── matmul_q5_0_core.v            ← Q5_0 2-core block-streaming core
│   ├── matmul_q4k_core.v             ← Q4_K block-decode core
│   ├── matmul_q6_k_core.v            ← Q6_K block-decode core
│   ├── matmul_int16_core.v           ← INT16 general core
│   ├── matmul_top.v                  ← PhaseB quad-core top (legacy)
│   ├── axihp_read_master.v           ← AXI HP read DMA
│   ├── axihp_write_master.v          ← AXI HP write DMA
│   ├── sim_ddr_axi_hp.v              ← DDR simulation model
│   ├── Makefile                      ← iverilog test targets
│   └── DESIGN.md                     ← RTL architecture details
├── sim/
│   ├── tmac_gguf.cpp                 ← Full C++ inference pipeline
│   ├── fpga_sim.hpp                  ← FPGA simulator + PhaseB descriptor model
│   ├── matmul_q8.cpp                 ← Q8_0 logits path wrapper
│   ├── test_integration.cpp          ← Integration tests
│   └── chat.py                       ← Chat interface
├── scripts/
│   ├── extract_tmac.py               ← GGUF → TMAC converter
│   ├── test_integration.sh           ← Test suite runner
│   └── verify_layers_fast.py         ← Layer verification
├── models/                           ← Model files (gitignored)
├── docs/                             ← Architecture docs + historical debug logs
├── firmware/                         ← ARM runtime (aspirational — implementation pending)
└── gguf-tools-main/                  ← Third-party GGUF inspection tool
```

## Hardware: HP FSM Register Map (0x43C00000)

| Offset | Name | Access | Description |
|--------|------|--------|-------------|
| 0x00 | REG_START | R/W | [0]: write 1 to start chain |
| 0x10 | REG_Q8_NUM_GROUPS | R/W | [3:0]: Q8 column groups (fallback) |
| 0x14 | REG_STATUS | R | [8]=rd_done, [9]=wr_done, [15]=busy |
| 0x18 | REG_DESC_BASE | R/W | Descriptor chain base DDR address |
| 0x1C | REG_DESC_TAIL | R/W | Write 1 to enable chain |
| 0x20 | REG_DESC_HEAD | R | Current descriptor index |
| 0x28 | REG_DEBUG | R | FSM state + bus status bits |
| 0x2C | REG_CLK_CNT | R | Free-running clock counter |
| 0x3C | REG_Q8_DEBUG | R | Q8 core state + counters |

## Documentation

- **[AGENTS.md](AGENTS.md)** — Full architecture, register map, descriptor protocol, FSM states, debug guide, hardware bringup history
- **[vivado_integration/API.md](vivado_integration/API.md)** — Hardware API reference: descriptors, DDR layouts, C++ usage
- **[verilog/DESIGN.md](verilog/DESIGN.md)** — RTL architecture: pipelines, memory, testbenches
- **[docs/architecture.md](docs/architecture.md)** — Model architecture, quantization formats

## References

- **Model**: [Qwen2-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct)
- **Quantization**: [llama.cpp GGUF](https://github.com/ggerganov/llama.cpp)
- **Board**: MicroPhase Z7-Lite (Zynq-7010, xc7z010clg400-1)
