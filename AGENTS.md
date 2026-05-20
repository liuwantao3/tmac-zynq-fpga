## Goal
- Implement Q4_K quantization support (Q4_K→INT16 dequant + INT16×INT16 matmul) in the FPGA accelerator, targeting Qwen 2 0.5B FFN layers (largest workload).

## Constraints & Preferences
- Must coexist with existing Q8_0 mode via mode switching (bit 6 of `reg_ctrl_user`).
- Same 64×64 tile structure and same 3-stage pipeline pattern.
- BRAM budget: 60 available, 1 used for Q8_0 → can use 2 for Q4_K.
- DSP budget: 8 of 80 used → no DSP increase for Q4_K core.
- Zynq 7010 target; AXI4-Lite memory-mapped I/O only (no direct DDR access).
- Q4_K weight data is written to AXI buffer addresses 0x1000-0x2FFF as pre-dequant INT16 bytes (8192 bytes/tile).
- AXI-Lite address ranges 0x2000-0x20FF (scale) and 0x2100-0x217F (act) overlap with weight range in Q4K mode — scale/act writes disabled when `mode_q4k` is active.

## Progress
### Done
- Analyzed Q4_K format (256 weights/block, 144 bytes/block: d+f16 + dmin+f16 + 12B scales + 128B qs).
- Committed current state as `pre_Q4 implementation` (git: c7703e9).
- Created `docs/Q4_K_IMPLEMENTATION_PLAN.md` (5-phase plan, moved to docs/ after implementation).
- Phase 1 C++ simulation: added `matmul_fpga_q4k()` to `tmac_gguf.cpp`, `axi_vecmul_tile_q4k()` to `fpga_sim.hpp`, `--fpga-q4k` flag.
- Fixed bug in `matmul_fpga_q4k()`: missing `row_scale_fp` division when quantizing tile weights to INT16.
- Verified: `--fpga-q4k` produces same output as `--fpga-int16` and CPU-only on Qwen 2 0.5B Q4_K_M model.
- Created `verilog/matmul_q4k_core.v` — Q4_K Verilog core with 512×128-bit wmem (2 BRAMs), INT16×INT16 pipeline, 3-stage FSM.
- Modified `verilog/matmul_top.v` (was `matmul_q8_top.v`): instantiated both cores, added `mode_q4k` muxing, expanded `weight_buf` to 8192 bytes.
- Fixed Q4K core `wt_addr` width 12→13 bits for 512-entry wmem.
- Fixed byte-lane addressing: `wt_addr[12:9]` = lane (0-15), `[8:0]` = entry (0-511).
- Updated AXI write address decode to cover 0x1000-0x2FFF.
- Fixed read path priority: scale/act before weight_buf (else-if cascade).
- Updated loading FSM for Q4K mode: 8192 bytes with `{load_addr[3:0], load_addr[12:4]}` addr format.
- Fixed weight_buf exclusion guard — bypassed in Q4K mode so scale/act address range (0x2000-0x217F) writes to weight_buf.
- Fixed Verilog syntax errors: `mode_q4k` forward reference, `reg`→`wire` for `assign`-driven signals.
- Created `tb_matmul_q4k.v` — standalone Q4K testbench with 7 directed tests (all-ones, zero, negative, col-varying, mixed per-row reference).
- Created `test_integration.cpp` — C++ integration test exercising INT16+Q8+Q4K-AXILITE paths vs CPU reference.
- Created `scripts/test_integration.sh` — shell script orchestrating full test suite.
- Added C++ AXI-Lite buffer API: `AxiliteAccelState`, `axilite_write_buf()`, `axilite_q4k_run()`, `axi_vecmul_tile_q4k_axilite()`.
- Fixed `axilite_write_buf`: incorrect `(data >> 8) & 0xFF00` mask → `((data >> 8) & 0xFF) << 8` for 16-bit buffer writes.
- Fixed `axilite_write_buf`: act_buf/scale_buf now write two 16-bit entries per 32-bit word (matching Verilog).
- Fixed `axilite_write_buf`: scale/act writes gated with `!mode_q4k` to prevent corruption when weight data writes through the same addresses.
- Fixed Verilog `write_buffer` scale/act writes gated with `!mode_q4k` for same reason.
- **Fixed act_buf corruption bug**: removed act_buf write path from `axilite_write_buf` (the act path fired for both activation AND weight writes to 0x2100-0x217F, corrupting act_buf with weight data). `axi_vecmul_tile_q4k_axilite` now writes act_buf directly.
- **Test 2 (Q4K AXI-Lite compute) PASS**: all 64 rows match reference.
- **Test 3 (5 random tiles)**: Q4K-AXILITE passes all tiles; 11 Q8 failures are pre-existing (unrelated to Q4K work).
- Fixed Test 1 address map test pattern (byte-lane overflow bug: `(off+b)<<(b*8)` overflows when off+b>255; replaced with fixed `0x03020100` pattern).
- **Fixed wmem address mapping bug** (`matmul_q4k_core.v:68`): core read used `{g, k}` = g*64+k (group-major), but sequential loading writes col-major (entry 0 = col 0 rows 0..7). Changed to `{k, g}` = k*8+g to match loading. Tests 1-5 (constant weights) passed despite bug; Tests 6-7 (varying) now pass.
- **Fixed `load_addr` overflow** (`tb_matmul_q4k.v:58`): `reg [12:0]` wrapped before 8192; infinite load loop. Changed to `integer`.
- **`tb_matmul_q4k.v`: ALL 7 TESTS PASSED** (all-ones, zero, negative, col-varying, per-row reference) — 448/448 checks pass.
- **C++ integration tests: ALL 3 TESTS PASSED** (addr map, Q4K compute, 5-tile roundtrip).
- **Verilog cleanup**: renamed `matmul_q8_top.v` → `matmul_top.v`, removed outdated files (`tb_minimal.v`, `tb_simple.v`, `tb_simple2.v`), rewrote `DESIGN.md` for dual-core, updated `Makefile` with Q4K targets + removed duplicate var defs.
- **`sim/fpga_sim.hpp` dead code removal**: removed `DspPhase`, `PipelineStats`, `AxiTiming`, `AxiTraceFn`, `axi_vecmul_tile_int8`, `read_q8_scale`, `compute_q8_row_scales`, `compute_combined_scales`, `fpga_logits_q8`, `AXI_WEIGHT_ADDR`, `AXI_ACT_ADDR`, old `axi_vecmul_tile_q4k` (CPU-dequant variant), tracer infrastructure (`set_tracer`/`trace_fn_`/`trace_cycle_`), `latency_ms_`/`set_latency_ms`, `interrupt()`, `add_cycles()`, `REG_A_ADDR_LO/B/C`, `CTRL_INT_ENABLE`, and several unused private members. Also removed unsupported `--dump-tiles` flags from Q4K/AXI-Lite help text.

### Remaining
- Add `--dump-tiles` support for the Q4K AXI-Lite path (needed for Verilog co-simulation in Phase 5).
- Run full `scripts/test_integration.sh` on model data (Phases 5-7).
- Fix pre-existing Q8 AXI-Lite path failure (all zeros in Test 3).

## Key Decisions
- **Pre-dequant in C++ approach**: Q4_K→INT16 dequant done on CPU, FPGA receives pre-dequantized INT16 data. Simplifies Verilog, validates INT16×INT16 pipeline first.
- **Dual-core instantiation**: Both `matmul_q8_core` and `matmul_q4k_core` coexist. Mode bit (`reg_ctrl_user[6]`) selects active core.
- **Expanded weight_buf**: 4096→8192 bytes with AXI address range 0x1000-0x2FFF. Overlaps with scale/act ranges at 0x2000-0x217F — resolved via `mode_q4k` gating (scale/act writes disabled in Q4K mode).
- **Byte-lane addressing for Q4K wmem**: `{byte_lane[3:0], entry[8:0]}` from load_addr[12:0] where byte_lane=load_addr%16 and entry=load_addr÷16. Uses 13-bit `wt_addr` (was 12-bit for Q8).
- **act_buf write isolation**: In Q4K mode, `axi_vecmul_tile_q4k_axilite` writes act_buf directly (not through `axilite_write_buf`) to avoid contamination by weight writes at the same address.

## Critical Context
- Qwen 2 0.5B `q4_k_m` model: 303 tensors, 374 MB. 24 layers, H=896, INTER=4864. FFN (gate+up+down) accounts for ~88% of matmul workload.
- Q4_K block: 144 bytes / 256 weights = 0.5625 B/weight vs Q8_0 at 1.0625 B/weight (47% compression).
- AXI-Lite buffer address map: `0x1000-0x1FFF` weight_low, `0x2000-0x20FF` scale, `0x2100-0x217F` act, `0x2200-0x2FFF` weight_high, `0x4000-0x41FF` result_lo, `0x4200-0x43FF` result_hi.
- Q4K wmem loading uses 16-byte granularity (2×64-bit halves per entry) mapped to 8 INT16 values per wmem entry.
- Q4K wmem address mapping: core reads with `{k, g}` = k*8+g (col-major). Loading writes tile data col-major: entry 0 = col 0 rows 0..7, entry 8 = col 1 rows 0..7.
- Both Verilog (iverilog modules compile clean) and C++ (g++ -std=c++14) compile.
- Integration test: Test 1 (addr map) PASS, Test 2 (Q4K compute) PASS, Test 3 (Q4K-AXILITE roundtrip) PASS on 5 random tiles; Q8 path has pre-existing zeros.
- Q4K Verilog testbench: ALL 7 TESTS PASSED (448/448 checks).

## Relevant Files
- `/Users/arctic/fpga/docs/Q4_K_IMPLEMENTATION_PLAN.md`: Implementation plan (archived post-implementation)
- `/Users/arctic/fpga/verilog/matmul_q4k_core.v`: Q4_K core (512×128-bit wmem, INT16×INT16 pipeline, 3-stage FSM)
- `/Users/arctic/fpga/verilog/matmul_top.v`: Top module — dual-core, `mode_q4k`, expanded weight_buf, Q4K loading FSM
- `/Users/arctic/fpga/verilog/tb_matmul_q4k.v`: Q4_K testbench (7 directed tests — ALL PASS)
- `/Users/arctic/fpga/verilog/Makefile`: Makefile for Verilog simulation (needs Q4K targets)
- `/Users/arctic/fpga/sim/fpga_sim.hpp`: AXI-Lite buffer API + Q4K simulation functions
- `/Users/arctic/fpga/sim/tmac_gguf.cpp`: Full inference pipeline, FPGA backends, dispatch
- `/Users/arctic/fpga/sim/test_integration.cpp`: C++ integration test (ALL 3 TESTS PASS)
- `/Users/arctic/fpga/scripts/test_integration.sh`: Shell orchestration script for full test suite
```

