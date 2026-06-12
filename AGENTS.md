## Project State

Qwen2-0.5B FPGA accelerator targeting Zynq 7010. Quad-core Verilog RTL: INT16×INT16 (general), Q8_0 dequant (logits), Q5_0 block decode (attn_q/k/o, ffn_gate/up), Q6_K block decode (ffn_down even), Q4_K block decode (ffn_down odd), with AXI4-Lite memory-mapped I/O.

**Model is Q4_K_M (mixed quantization).** Actual tensor types:

| tensor | shape | type | C++ path | Verilog core |
|--------|-------|------|----------|-------------|
| `token_embd` | 151936×896 | Q8_0 | ✅ | ✅ `matmul_q8_core.v` |
| `attn_v` | 128×896 | Q8_0 | ✅ | ✅ `matmul_q8_core.v` |
| `attn_q`, `attn_k`, `attn_output` | 896×896 | Q5_0 | ✅ `matmul_fpga_q5_0` | ✅ `matmul_q5_0_core.v` |
| `ffn_gate`, `ffn_up` | 4864×896 | Q5_0 | ✅ `matmul_fpga_q5_0` | ✅ `matmul_q5_0_core.v` |
| `ffn_down` (layers 0,2,4,...) | 896×4864 | Q6_K | ✅ `matmul_fpga_q6_k` | ✅ `matmul_q6_k_core.v` |
| `ffn_down` (layers 1,3,5,...) | 896×4864 | Q4_K | ✅ `matmul_fpga_q4_k` | ✅ `matmul_q4k_core.v` (56×256 tile) |

## Key Decisions (2026-05-22)

1. **Removed `matmul_q4k_2x896_core.v`** — tile [2×896] was a misunderstanding of GGUF Q4_K format. Q4_K tensor stores rows sequentially with block stride = cols/256, not contiguously by 2 rows.

2. **Single Q4_K core with 56×256 tile** — tile = 56 rows × 256 cols = 56 blocks × 144B = 8064 bytes. Fits in 8192 weight_buf. Used for ffn_down on odd layers (1,3,5,...).

3. **Q5_0 tile = 8×896** — 8 rows × 896 cols = 224 blocks × 22B = 4928 bytes. Fits in 8192 weight_buf. Used for attn_q/k/o and ffn_gate/up.

4. **Q6_K tile = 32×256** — 32 rows × 256 cols = 32 blocks × 210B = 6720 bytes. Fits in 8192 weight_buf. Used for ffn_down on even layers (0,2,4,...).

5. **Q8_0 tile = 64×896** — 64 rows × 896 cols = 4100 bytes. Used for token_embd, attn_v, logits.

6. **INT16 fallback** — tiles that don't fit the above paths (e.g. F32 norms, any other tensor types) fall back to `matmul_fpga_int16` which uses pure INT16×INT16 via `MatmulAccel`.

7. **Vivado 2023.1** — development on Windows with Vivado 2023.1 (upgraded from 2019.2). LLVM clang 7.0 bundled with Vivado used as ARM cross-compiler (`--target=armv7a-none-eabi`). Vitis 2023.1 used for debugging/running.

## Architecture Summary

### User FSM flow (matmul_top.v):
```
IDLE → LOAD_WEIGHT → LOAD_ACT → COMPUTE → DRAIN → IDLE
```

### CPU-OP Descriptor Protocol (CPU/FPGA synchronization):

To handle CPU-only operations (RMSNorm, RoPE, SoftMax, bias add, SwiGLU, residual add) between FPGA matmuls, the descriptor chain supports a special `CPU_OP` descriptor type (`tensor_type = 15`).

When the FSM encounters a CPU_OP descriptor:
1. It sets `reg_chain_ctrl[2]=1` and pulses `desc_irq` → CPU interrupt fires
2. FSM enters `PH_CPU_OP_WAIT` state and **pauses** until CPU resumes it
3. CPU reads `reg_status` (status=3 = chain busy) to distinguish from chain-complete
4. CPU reads `reg_desc_head` to identify which descriptor index triggered the CPU_OP
5. CPU does its work (norm, softmax, etc.) and writes results to DDR
6. CPU clears `reg_isr[0]` (write REG_ISR) and writes `CHAIN_CTRL[0]=1` to resume
7. FSM clears the resume signal, advances to next descriptor

The CPU knows what operation to perform from the descriptor's position in the chain (the CPU built the chain, so it has an internal mapping: "descriptor 0 → attn_norm, descriptor 4 → bias+rope+softmax").

**CPU operations per layer** (from `tmac_gguf.cpp`):

| # | Operation | What it does | DDR I/O |
|---|-----------|-------------|---------|
| 1 | attn_norm | RMSNorm(hidden) | Read hidden, Write norm_out |
| 2 | q_bias | q += bias | Read q, Read bias, Write q |
| 3 | k_bias | k += bias | Read k, Read bias, Write k |
| 4 | v_bias | v += bias | Read v, Read bias, Write v |
| 5 | RoPE | apply_rope(q, k, pos) | Read q/k, Write q/k |
| 6 | SoftMax | score = q·k/√d, softmax, Σv·score | Read q/k/v + KV cache, Write context |
| 7 | residual | hidden += attn_out | Read hidden, Read attn_out, Write hidden |
| 8 | ffn_norm | RMSNorm(hidden) | Read hidden, Write norm2 |
| 9 | SwiGLU | silu(gate) * up | Read gate/up, Write swiglu_out |
| 10 | residual | hidden += ffn_out | Read hidden, Read ffn_out, Write hidden |

**Typical descriptor chain for one layer** (adjacent CPU ops batched):

```
Desc  0: CPU_OP           → attn_norm                          (result: norm_out)
Desc  1: matmul_q5_0      → attn_q     (act: norm_out)          (result: q)
Desc  2: matmul_q5_0      → attn_k     (act: norm_out)          (result: k)
Desc  3: matmul_q8_0      → attn_v     (act: norm_out)          (result: v)
Desc  4: CPU_OP           → bias+rope+softmax                   (result: context)
Desc  5: matmul_q5_0      → attn_output (act: context)          (result: attn_out)
Desc  6: CPU_OP           → residual+ffn_norm                   (result: norm2)
Desc  7: matmul_q5_0      → ffn_gate   (act: norm2)             (result: gate)
Desc  8: matmul_q5_0      → ffn_up     (act: norm2)             (result: up)
Desc  9: CPU_OP           → swiglu                              (result: swiglu_out)
Desc 10: matmul_q6k/q4k   → ffn_down   (act: swiglu_out)        (result: ffn_out)
Desc 11: CPU_OP           → residual                            (result: hidden)
```

12 descriptors × 28 layers = 336 descriptors per token. Each CPU_OP triggers one interrupt.
Adjacent CPU ops are batched into single descriptors (e.g. bias+rope+softmax = one CPU_OP).
Post-logits (softmax for sampling) is handled outside the descriptor chain.

### Descriptor Chain (OP→OP, no CPU):
Descriptors in DDR form a linked list. Each descriptor contains weight/act/result addresses, tensor type, and `act_total_bytes`. When descriptor N's `result_addr` equals descriptor N+1's `act_addr`, the chain auto-derives `act_total_bytes = prev.tile_res_rows × 8` (all cores output 48-bit fixed-point, 8 bytes/row). Header validation in `tb_phaseb.v` checks this invariant.

### Dispatch Logic (tmac_gguf.cpp:452-465):
```
if (g_fpga_q5_0 && A->type == TENSOR_Q5_0) → matmul_fpga_q5_0()  (attn_q/k/o, ffn_gate/up)
else if (g_fpga_q6_k && A->type == TENSOR_Q6_K) → matmul_fpga_q6_k()  (ffn_down even)
else if (g_fpga_q4k && A->type == TENSOR_Q4_K) → matmul_fpga_q4_k()  (ffn_down odd)
else if (g_fpga_q8 && A->type == TENSOR_Q8_0) → matmul_fpga_q8()   (token_embd, attn_v, logits)
else → matmul_fpga_int16()   (F32 norms, fallback)
```

### Tile Sizes and Buffer Usage:

| Type | Tile | Blocks | Bytes/tile | weight_buf | Result Bytes/row |
|------|------|--------|------------|------------|-----------------|
| Q8_0 | 64×896 | — | 4100 | 4096 | 8 |
| Q5_0 | 8×896 | 224 | 4928 | 8192 | 8 |
| Q6_K | 32×256 | 32 | 6720 | 8192 | 8 |
| Q4_K | 56×256 | 56 | 8064 | 8192 | 8 |
| INT16 | 64×64 | — | 8192 | 8192 | 8 |

All cores output S24.8 fixed-point (48-bit accumulator, zero-extended to 64-bit in DDR).

### Existing Verilog Cores:

| Core | Tile | Cycle/tile | Status |
|------|------|-----------|--------|
| `matmul_q8_core.v` | 64×896 | ~515 | ✅ Working |
| `matmul_q4k_core.v` | 56×256 | ~? | ✅ Working |
| `matmul_int16_core.v` | 64×64 | 515 | ✅ Working |
| `matmul_top.v` | — | — | ✅ 5 cores instantiated (Q8, Q4K, Q5_0, Q6_K, INT16) |

### Missing Verilog Cores:

| Core | Tile | Status |
|------|------|--------|
| None | — | All cores implemented ✅ |

## Remaining Work

1. ~~Implement `matmul_q5_0_core.v`~~ ✅ Done
2. ~~Implement `matmul_q6_k_core.v`~~ ✅ Done
3. ~~Instantiate Q5_0 and Q6_K cores in `matmul_top.v`~~ ✅ Done (2026-05-27)
4. ~~Create testbenches for Q5_0 and Q6_K cores~~ ✅ Done (2026-05-27)
5. Vivado simulation — run with Vivado 2019
6. ~~End-to-end inference test~~ ✅ Verified: all paths produce same token (11 358 3003)
7. ~~Descriptor chain OP→OP format fix~~ ✅ Done (2026-05-27): auto-derive `act_total_bytes` from previous result when chaining

## Build & Run Commands

```bash
# Verilog tests (iVerilog)
make -C verilog all                     # Q8 + Q4K + Q5_0 + INT16 smoke

# Vivado batch build
C:\Xilinx\Vivado\2023.1\bin\vivado.bat -mode batch -source vivado_integration/build_bd.tcl

# JTAG load via XSDB
C:\Xilinx\Vivado\2023.1\bin\xsdb.bat vivado_integration/sw/load.tcl

# C++ integration test
g++ -std=c++14 -O2 -I sim -I gguf -I . sim/test_integration.cpp -lpthread -o /tmp/ti
/tmp/ti

# Build inference engine
g++ -std=c++17 -pthread -O2 -I sim -I gguf -I . sim/tmac_gguf.cpp sim/matmul_q8.cpp -o sim/tmac_gguf

# FPGA simulation flags (individual paths can be combined)
echo "9707" | ./sim/tmac_gguf models/model.tmac --fpga-q8              # Q8_0 only
echo "9707" | ./sim/tmac_gguf models/model.tmac --fpga-q5-0           # Q5_0 only
echo "9707" | ./sim/tmac_gguf models/model.tmac --fpga-q6-k          # Q6_K only
echo "9707" | ./sim/tmac_gguf models/model.tmac --fpga-q4k          # Q4_K only
echo "9707" | ./sim/tmac_gguf models/model.tmac --fpga-q5-0 --fpga-q6-k --fpga-q4k  # combined
echo "9707" | ./sim/tmac_gguf models/model.tmac --fpga-q8 --fpga-q5-0 --fpga-q6-k --fpga-q4k  # all paths

# Full test suite
bash scripts/test_integration.sh

# Model conversion
python3 scripts/extract_tmac.py models/qwen2-0_5b-instruct-q4_k_m.gguf /tmp/model.tmac
```

## File Inventory

### Verilog RTL
- `verilog/matmul_top.v` — Quad-core top: AXI4-Lite slave (inline, replaces orphan axilite_slave.v), 8192-byte weight_buf, loading FSM, mode mux (Q8/Q4K/Q5_0/Q6_K/INT16)
- `verilog/matmul_q8_core.v` — Q8_0 compute core: 512×64-bit wmem, dequant LUT, 3-stage FSM
- `verilog/matmul_q4k_core.v` — Q4_K block decode: 2304-byte block buffer, S24.8 fixed-point, 56×256 tile
- `verilog/matmul_q5_0_core.v` — Q5_0 block decode: 8×896 tile, 224 blocks/tile, row_scale normalization
- `verilog/matmul_q6_k_core.v` — Q6_K block decode: 32×256 tile, 32 blocks/tile, super_scale + per-sub-block scales
- `verilog/matmul_int16_core.v` — General INT16×INT16 core: 512×128-bit wmem, 3-stage FSM
- `verilog/dequant_lut.v` — Q8_0 dequant ROM (standalone, not instantiated)
- `verilog/systolic_8x8.v` — 8×8 systolic array (standalone, not used)

### Verilog Testbenches
- `verilog/tb_matmul_q8.v` — Q8 core tests
- `verilog/tb_matmul_q4k.v` — Q4K core tests
- `verilog/tb_minimal_q4k.v` — Q4K smoke test
- `verilog/tb_int16_smoke.v` — INT16 smoke test
- `verilog/tb_cosim.v` — Q8_0 co-simulation
- `verilog/tb_cosim_q4k.v` — Q4_K co-simulation
- `verilog/tb_matmul_q5_0.v` — Q5_0 core tests (fabricated patterns)
- `verilog/tb_matmul_q6_k.v` — Q6_K core tests (fabricated patterns)
- `verilog/tb_cosim_q5_0.v` — Q5_0 co-simulation (waits for tile dump)
- `verilog/tb_cosim_q6_k.v` — Q6_K co-simulation (waits for tile dump)

### C++ Simulation
- `sim/tmac_gguf.cpp` — Full inference pipeline, dispatch by tensor type
- `sim/fpga_sim.hpp` — `MatmulAccel`, `axi_vecmul_tile_*` functions, decode logic
- `sim/matmul_q8.cpp` — Q8_0 logits path wrapper
- `sim/test_integration.cpp` — Integration tests
- `sim/chat.py` — Chat interface

### Scripts
- `scripts/extract_tmac.py` — GGUF→TMAC converter
- `scripts/ground_truth_v2.py` — Ground truth generation
- `scripts/verify_layers_fast.py` — Layer verification
- `scripts/design_iteration.sh` — Vivado iteration loop
- `scripts/feedback_parser.py` — Report parser
- `scripts/test_integration.sh` — Test suite

### Documentation
- `verilog/DESIGN.md` — Architecture, timing
- `docs/architecture.md` — Model, quantization formats
- `docs/FPGA_PERFORMANCE_ANALYSIS.md` — Performance analysis
- `docs/Q4_K_IMPLEMENTATION_PLAN.md` — Original plan (outdated)

## Target Board: MicroPhase Z7-Lite

- **Part**: `xc7z010clg400-1` (Zynq-7010, 28nm, 432 CLBs, 240 DSP48E1, 4.9 Mb BRAM)
- **DDR3**: 512 MB, Micron **MT41J256M16 RE-125** (4 Gbit, 16-bit bus, 15 row × 10 col × 3 bank)
- **DDR speed**: DDR3-1066F (533 MHz core, CL=7, CWL=6, tRCD=7, tRP=7, tRAS=35ns, tRC=48.91ns, tFAW=40ns)
- **PS7 clock config**: 33.333 MHz crystal → ARM=666.667 MHz, DDR=533.333 MHz, PL=100 MHz
- **UART console**: UART0 (MIO 14/15) at 115200 baud
- **Debug**: JTAG via Digilent HS-2 on the onboard FTDI/JTAG bridge
- **Peripherals**: No Ethernet, no SD card (bare-metal only)

**DDR address map** (from PS7 config): 0x0010_0000 to 0x2000_0000 (512 MB), CPU accesses at 0x0000_0000 alias after MMU/cache setup; bare-metal uses physical 0x0010_0000 base.

## Phase 1: AXI HP + INT16

### Batch Build Framework

Three scripts form the batch build/debug pipeline:

| Script | Purpose | Command |
|--------|---------|---------|
| `vivado_integration/build_bd.tcl` | Full Zynq BD (PS7 + AXI Lite + AXI HP) | `vivado -mode batch -source build_bd.tcl` |
| `vivado_integration/build_debug.tcl` | Minimal test design with ILA | `vivado -mode batch -source build_debug.tcl` |
| `vivado_integration/capture_ila.tcl` | ILA capture (WIP) | `vivado -mode batch -source capture_ila.tcl` |

### Batch Build with ILA (`build_debug.tcl`) — Step by Step

```tcl
# 1. Create project for xc7z010
create_project -force debug_test $proj_dir -part xc7z010clg400-1
# 2. Add RTL source
add_files -fileset sources_1 [file normalize "$rtl_dir/debug_test.v"]
set_property top debug_test [current_fileset]
# 3. Synthesize
launch_runs synth_1 -jobs 8
wait_on_run synth_1
# 4. Open synthesized design, create ILA core
open_run synth_1
create_debug_core u_ila_0 ila
set_property C_DATA_DEPTH 1024 [get_debug_cores u_ila_0]
set_property C_TRIGIN_EN false [get_debug_cores u_ila_0]
set_property C_TRIGOUT_EN false [get_debug_cores u_ila_0]
# 5. Connect ILA clock to BUFG output (routable)
connect_debug_port u_ila_0/clk [get_nets {clk_IBUF_BUFG}]
# 6. Connect probes to MARK_DEBUG nets
connect_debug_port u_ila_0/probe0 [get_nets {dbg_tick}]
# 7. Write synth checkpoint
write_checkpoint
# 8. Implement with relaxed DRC (unconstrained I/O test design)
set_property STEPS.write_bitstream.TCL.PRE {write_bitstream_hook.tcl} [get_runs impl_1]
launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1
```

### ILA Capture (`capture_ila.tcl`) — Step by Step

```tcl
# 1. Connect to hardware
open_hw_manager
connect_hw_server
open_hw_target
current_hw_device [lindex [get_hw_devices] end]
# 2. Set bitstream and probes file
set_property PROGRAM.FILE $bit [current_hw_device]
set_property PROBES.FILE $ltx [current_hw_device]
# 3. Program FPGA
program_hw_devices [current_hw_device]
# 4. Configure BSCAN scan chain (required for ILA detection)
set_property BSCAN_SWITCH_USER_MASK 0x02 [current_hw_device]
refresh_hw_device [current_hw_device]
# 5. Find ILA and set depth
set ila [lindex [get_hw_ilas -of_objects [current_hw_device]] 0]
set_property CONTROL.DATA_DEPTH 1024 $ila
# 6. Arm ILA and capture
run_hw_ila $ila
after 2000
# 7. Read data and save to CSV
set data [read_hw_ila_data $ila]
write_hw_ila_data -force -csv_file ila_data.csv $data
```

### Known Issues

| Issue | Cause | Workaround |
|-------|-------|------------|
| AXI HP read stuck in RD_WT | HP interconnect/auto_pc | Use ILA to capture ARADDR/ARVALID/ARREADY |
| ILA not detected after program | BSCAN_SWITCH_USER_MASK not set | Set 0x02 and refresh |
| xsdb can't find ARM core | DAP error after Vivado program | Run ps7_init from xc7z010 target |
| ps7_init.tcl path | File at `vivado_integration/` not `proj_bd/` | Use absolute path |

### Verification Scripts

| Script | What it does |
|--------|-------------|
| `sw/hp_test.c` | Full HP test: writes weights/acts to DDR, starts computation, checks results |
| `sw/run_hp.tcl` | XSDB: init PS7, load ELF, run |
| `sw/build.tcl` | XSCT: create Vitis platform + app + build |
| `vivado_integration/check_props.tcl` | Debug: list all hw_device properties |

### Relevant Files

- `vivado_integration/build_bd.tcl`: Vivado batch build script (PS7 DDR=MT41J256M16, UART0, 100 MHz PL, AXI interconnect, bitstream + XSA)
- `vivado_integration/rtl/axi_wrap_int16.v`: AXI4-Lite wrapper with BRAM weight_buf, DSP48 compute
- `vivado_integration/sw/main.c`: Bare-metal INT16 matmul test
- `vivado_integration/sw/regs.h`: Register map (IP_BASE=0x43C00000, UART0=0xE0000000)
- `vivado_integration/sw/uart.c`: UART0 polling driver (115200 baud, 50 MHz ref clock)
- `vivado_integration/proj_bd/`: Vivado project outputs (post_synth.dcp, post_impl.dcp, matmul_bd.xsa, system_wrapper.bit)
- `verilog/matmul_int16_core.v`: INT16 compute core (standalone verified, 8 DSP, 2 BRAM)
- `verilog/matmul_top.v`: Quad-core top (Q8, Q4K, Q5_0, Q6_K, INT16)
