# Vivado Block Design Integration

Files needed from the Mac:
- `rtl/axi_wrap_int16.v`   — AXI4-Lite INT16 wrapper (top module)
- `rtl/matmul_int16_core.v` — INT16 matmul core (instantiated by wrapper)
- `sw/*`                     — Bare-metal PS7 C test code

## Vivado Flow

### 1. Create Project
- Launch Vivado → Create Project → RTL Project
- Target: `xc7z010clg400-1`
- Do NOT specify sources yet (or add them now)

### 2. Add RTL Sources
- Add `axi_wrap_int16.v` and `matmul_int16_core.v` as Verilog files

### 3. Create Block Design
- Create Block Design → name e.g. `system`
- Add IP: `ZYNQ7 Processing System`
- Run Block Automation (accept defaults, or configure DDR/UART for your board)
- Add IP: `AXI Interconnect` (1 master, 1 slave)
- Run Connection Automation: connect PS7 M_AXI_GP0 → AXI Interconnect S00_AXI
- Add our RTL module: Right-click in block design → Add Module → select `axi_wrap_int16`
- Run Connection Automation: this will connect AXI Interconnect M00_AXI → our module's S_AXI
  - If not auto-detected: right-click our module → Create Interface → AXI4Lite Slave → name `S_AXI` → map ports manually
- Connect clk/rst_n: right-click → Make External, or connect to PS7 FCLK_CLK0 / FCLK_RESET0_N

### 4. Address Map
Vivado will assign a base address for our module (typically `0x43C0_0000`). This must match `IP_BASE` in `sw/regs.h`.

### 5. Generate Bitstream
- Right-click block design → Generate Output Products
- Create HDL Wrapper (top = wrapper)
- Run Synthesis → Implementation → Generate Bitstream

### 6. Export to Vitis
- File → Export → Export Hardware (include bitstream)
- File → Launch Vitis
- Create Application Project → import our `sw/` files
- Build, run via JTAG

## Address Map (axi_wrap_int16)

| Address range | Content |
|---|---|
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

## Test Weights

The test in `sw/main.c` uses:
- W[col][row] = row*64 + col (counting pattern)
- Act[col] = col + 1 (1..64)
- Expected result[row] = row*133120 + 87360

The golden reference is computed on-chip for self-verification.
