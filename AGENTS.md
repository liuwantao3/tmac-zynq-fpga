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

7. **Vivado 2019** — development on Windows with Vivado 2019 (previously used iVerilog on Mac).

## Architecture Summary

### User FSM flow (matmul_top.v):
```
IDLE → LOAD_WEIGHT → LOAD_ACT → COMPUTE → DRAIN → IDLE
```

### Dispatch Logic (tmac_gguf.cpp:452-465):
```
if (g_fpga_q5_0 && A->type == TENSOR_Q5_0) → matmul_fpga_q5_0()  (attn_q/k/o, ffn_gate/up)
else if (g_fpga_q6_k && A->type == TENSOR_Q6_K) → matmul_fpga_q6_k()  (ffn_down even)
else if (g_fpga_q4k && A->type == TENSOR_Q4_K) → matmul_fpga_q4_k()  (ffn_down odd)
else if (g_fpga_q8 && A->type == TENSOR_Q8_0) → matmul_fpga_q8()   (token_embd, attn_v, logits)
else → matmul_fpga_int16()   (F32 norms, fallback)
```

### Tile Sizes and Buffer Usage:

| Type | Tile | Blocks | Bytes/tile | weight_buf |
|------|------|--------|------------|------------|
| Q8_0 | 64×896 | — | 4100 | 4096 |
| Q5_0 | 8×896 | 224 | 4928 | 8192 |
| Q6_K | 32×256 | 32 | 6720 | 8192 |
| Q4_K | 56×256 | 56 | 8064 | 8192 |
| INT16 | 64×64 | — | 8192 | 8192 |

### Existing Verilog Cores:

| Core | Tile | Cycle/tile | Status |
|------|------|-----------|--------|
| `matmul_q8_core.v` | 64×896 | ~515 | ✅ Working |
| `matmul_q4k_core.v` | 56×256 | ~? | ✅ Working |
| `matmul_int16_core.v` | 64×64 | 515 | ✅ Working |
| `matmul_top.v` | — | — | ✅ 3 cores instantiated (Q8, Q4K, INT16) |

### Missing Verilog Cores:

| Core | Tile | Status |
|------|------|--------|
| None | — | All cores implemented ✅ |

## Remaining Work

1. ~~Implement `matmul_q5_0_core.v`~~ ✅ Done
2. ~~Implement `matmul_q6_k_core.v`~~ ✅ Done
3. Instantiate Q5_0 and Q6_K cores in `matmul_top.v` — add mode bits and mux
4. Create testbenches for Q5_0 and Q6_K cores
5. Vivado simulation — run with Vivado 2019
6. ~~End-to-end inference test~~ ✅ Verified: all paths produce same token (11 358 3003)

## Build & Run Commands

```bash
# Verilog tests (iVerilog)
make -C verilog all                     # Q8 + Q4K + Q5_0 + INT16 smoke

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
- `verilog/matmul_top.v` — Quad-core top: AXI4-Lite slave, 8192-byte weight_buf, loading FSM, mode mux (Q8/Q4K/Q5_0/Q6_K/INT16)
- `verilog/matmul_q8_core.v` — Q8_0 compute core: 512×64-bit wmem, dequant LUT, 3-stage FSM
- `verilog/matmul_q4k_core.v` — Q4_K block decode: 2304-byte block buffer, S24.8 fixed-point, 56×256 tile
- `verilog/matmul_q5_0_core.v` — Q5_0 block decode: 8×896 tile, 224 blocks/tile, row_scale normalization
- `verilog/matmul_q6_k_core.v` — Q6_K block decode: 32×256 tile, 32 blocks/tile, super_scale + per-sub-block scales
- `verilog/matmul_int16_core.v` — General INT16×INT16 core: 512×128-bit wmem, 3-stage FSM
- `verilog/axilite_slave.v` — AXI4-Lite slave + register file
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