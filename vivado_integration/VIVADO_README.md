# Vivado Integration Guide

This directory contains the bare-metal PS7 software and integration docs for the FPGA accelerator.

## Two Design Variants

### Variant A: INT16-Only (`axi_wrap_int16.v`)
Simple single-core design. AXI4-Lite only, no HP ports. Quick to synthesize.

### Variant B: Quad-Core (`matmul_top.v`)
Full quad-core design with all quantization paths. Includes AXI HP ports for DDR access.
This is the **production design** used in the descriptor chain OP→OP flow.

---

## Quick Start: Variant A (INT16-Only)

### Files
- `rtl/axi_wrap_int16.v` — AXI4-Lite wrapper (top module)
- `rtl/matmul_int16_core.v` — INT16 matmul core (instantiated by wrapper)
- `sw/` — Bare-metal PS7 C test code

### Vivado Flow

#### 1. Create Project
```
Vivado → Create Project → RTL Project
Target: xc7z010clg400-1
Do NOT add sources yet
```

#### 2. Add RTL Sources
Add `axi_wrap_int16.v` and `matmul_int16_core.v` as Verilog files.

#### 3. Create Block Design
```
Create Block Design → name: system
Add IP → ZYNQ7 Processing System
Run Block Automation (accept defaults)
Add IP → AXI Interconnect (1 master, 1 slave)
Run Connection Automation
  → PS7 M_AXI_GP0 → AXI Interconnect S00_AXI
  → AXI Interconnect M00_AXI → axi_wrap_int16 S_AXI
Connect clk → FCLK_CLK0, rst_n → FCLK_RESET0_N
Make External: clk, rst_n, interrupt
```

#### 4. Address Map
Vivado assigns base address (e.g. `0x43C00000`). Record this and update `IP_BASE` in `sw/regs.h`.

#### 5. Generate Bitstream
```
Right-click block design → Generate Output Products
Create HDL Wrapper (top = wrapper)
Run Synthesis → Implementation → Generate Bitstream
```

#### 6. Export to Vitis
```
File → Export → Export Hardware (include bitstream)
File → Launch Vitis
Import `sw/` files, build, run via JTAG
```

### AXI4-Lite Address Map (INT16)

| Address | Content |
|---------|---------|
| `BASE + 0x0000` | AP_CTRL (write bit 0=start) |
| `BASE + 0x0004` | GIE |
| `BASE + 0x0008` | IER |
| `BASE + 0x000C` | ISR |
| `BASE + 0x0010` | CTRL_USER |
| `BASE + 0x0014` | STATUS (0=idle, 1=loading, 2=compute) |
| `BASE + 0x1000-0x107C` | Activations (64 × INT16) |
| `BASE + 0x2000-0x3FFF` | Weights (2048 × 32-bit = 8192 bytes) |
| `BASE + 0x4000-0x40FC` | Result lo (64 × 32-bit) |
| `BASE + 0x4200-0x427C` | Result hi (64 × 16-bit) |
| `BASE + 0x5000-0x507C` | Act readback |

### Test Weights
`sw/main.c` uses:
- `W[col][row] = row*64 + col` (counting pattern)
- `Act[col] = col + 1` (1..64)
- Expected: `result[row] = row*133120 + 87360`

---

## Full Design: Variant B (Quad-Core matmul_top.v)

The quad-core design is located at `verilog/matmul_top.v`. It instantiates all 5 cores:
- `matmul_int16_core` — INT16 fallback
- `matmul_q8_core` — Q8_0 path (token_embd, attn_v, logits)
- `matmul_q4k_core` — Q4_K path (ffn_down odd layers)
- `matmul_q5_0_core` — Q5_0 path (attn_q/k/o, ffn_gate/up)
- `matmul_q6_k_core` — Q6_K path (ffn_down even layers)

Plus AXI HP masters for DDR access (`axihp_read_master.v`, `axihp_write_master.v`).

### Files for Quad-Core Vivado Flow

Copy these from the repo to your Windows/Vivado machine:

**RTL (from `verilog/`):**
- `matmul_top.v`
- `matmul_int16_core.v`
- `matmul_q8_core.v`
- `matmul_q4k_core.v`
- `matmul_q5_0_core.v`
- `matmul_q6_k_core.v`
- `axihp_read_master.v`
- `axihp_write_master.v`
- `axilite_slave.v`

**RTL (from `oss-tools/workspace/rtl/`):**
- `synth_matmul_top.v` — AXI4-Lite wrapper for matmul_top (stubs HP ports)

### Vivado Flow: Quad-Core

#### 1. Create Project
```
Vivado → Create Project → RTL Project
Target: xc7z010clg400-1
```

#### 2. Add RTL Sources
Add all Verilog files above as design sources (do NOT add testbenches).

#### 3. Create Block Design
```
Create Block Design → name: system
Add IP → ZYNQ7 Processing System
Run Block Automation
Add IP → AXI Interconnect (1 master, 1 slave) — or 2 masters if using both HP ports
Run Connection Automation:
  → PS7 M_AXI_GP0 → AXI Interconnect S00_AXI
  → AXI Interconnect M00_AXI → synth_matmul_top S_AXI (AXI4-Lite)
  → Connect clk → FCLK_CLK0, rst_n → FCLK_RESET0_N
Make External: all ports on synth_matmul_top except S_AXI
```

For HP ports (DDR access), either:
- **Option 1**: Connect HP0/HP1 directly to ZYNQ if using AXI HP without a full interconnect
- **Option 2**: Stub HP ports in synth_matmul_top and connect to test logic

#### 4. Address Map
```
S_AXI (AXI4-Lite slave): base address e.g. 0x43C00000
HP0: base address 0x00000000 (DDR, 512 MB)
HP1: base address 0x20000000 (DDR, 512 MB)
```

#### 5. Generate Bitstream
```
Run Synthesis → Implementation → Generate Bitstream
```

### AXI4-Lite Register Map (matmul_top)

| Address | Name | Access | Description |
|---------|------|--------|-------------|
| 0x0000 | AP_CTRL | R/W | [0]=start, [1]=done, [2]=idle, [3]=ready |
| 0x0004 | GIE | R/W | Global Interrupt Enable |
| 0x0008 | IER | R/W | Interrupt Enable |
| 0x000C | ISR | R/W | Interrupt Status (W1C) |
| 0x0010 | CTRL_USER | R/W | Mode: [4]=INT16, [5]=Q8_0, [6]=Q4_K, [7]=Q5_0, [8]=Q6_K |
| 0x0014 | STATUS | R | 0=IDLE, 1=RUNNING, 2=DONE, 3=CHAIN_BUSY |
| 0x0018 | DESC_BASE | R/W | DDR base address of descriptor chain |
| 0x001C | DESC_TAIL | R/W | Number of descriptors |
| 0x0020 | DESC_HEAD | R | Current descriptor index |
| 0x0024 | CHAIN_CTRL | R/W | [0]=enable, [1]=reset, [2]=done |

### Descriptor Chain Format (32 bytes, DDR)

| Offset | Field | Type | Description |
|--------|-------|------|-------------|
| 0x00 | next_desc_addr | uint32_t | Pointer to next descriptor (0 = end) |
| 0x04 | weight_addr | uint32_t | DDR offset of weight tile |
| 0x08 | act_addr | uint32_t | DDR offset of activation vector |
| 0x0C | result_addr | uint32_t | DDR offset to write results |
| 0x10 | num_tiles | uint16_t | Number of output tiles |
| 0x12 | tile_bytes | uint16_t | Bytes per weight tile |
| 0x14 | tensor_type | uint8_t | 0=INT16, 6=Q5_0, 8=Q8_0, 12=Q4_K, 14=Q6_K |
| 0x15 | tile_res_rows | uint8_t | Result rows per tile |
| 0x16 | flags | uint8_t | [0]=interrupt |
| 0x17 | act_total_bytes | uint16_t | Activation bytes; auto-derived = prev_rows×8 if chained |
| 0x19 | num_col_groups | uint8_t | Column groups (Q4_K) |

### OP→OP Chaining

When chaining descriptors (result of op N = activation of op N+1):
1. Set `desc[N+1].act_addr = desc[N].result_addr`
2. The C++ chain builder (`tmac_gguf.cpp`) auto-derives `act_total_bytes = prev.tile_res_rows × 8`
3. All cores output 48-bit fixed-point S24.8 → 8 bytes/row in DDR

### Software Build

```bash
cd vivado_integration/sw
make clean all
# Produces: test_int16.elf, test_int16.bin
```

JTAG load via Vitis or:
```bash
openocd -f board/digilent-zynq7.cfg -c "init; halt; load_image test_int16.bin 0x00100000; reg pc 0x00100000; resume"
```

---

## OSS CAD vs Vivado

| | OSS CAD (Yosys+nextpnr) | Vivado |
|--|------------------------|--------|
| INT16 standalone | ✅ Works | ✅ Works |
| Quad-core (matmul_top) | ❌ PnR multi-driver error | ✅ Works |
| AXI HP ports | Limited support | ✅ Full support |
| Bitstream | ✅ (with fasm2frames) | ✅ Production |

The quad-core `matmul_top.v` has two AXI FSMs (write/read) driving shared signals. OSS CAD's nextpnr correctly rejects this as multi-driven. **Vivado synthesis handles this correctly** and is the recommended flow for the full design.
