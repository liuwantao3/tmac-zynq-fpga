# Code Audit — Qwen2-0.5B FPGA Accelerator

> **NOTE (2026-07-07):** This audit dates from 2026-05-27. Many items have since
> been resolved: Q8 pipeline rewritten, BRAM conversions done, Q5_0 rewritten,
> descriptor chain proven on hardware. Items 6, 8, 9, 10, 12 may still be relevant.
> For current status see `AGENTS.md`.

**Date:** 2026-05-27  
**Scope:** `sim/tmac_gguf.cpp`, `sim/fpga_sim.hpp`, `verilog/` (all RTL)

---

## CRITICAL BUGS

### 1. Q4K weight write never connected (`matmul_top.v:180–194`)

`q4k_wt_we` is declared (`reg`) and connected to `u_core_q4k.wt_we`, but **never assigned** anywhere in `matmul_top.v`. The FSM's weight-loading section writes to the shared `wt_we`, `wt_addr`, `wt_din` signals (lines 776–786). All other cores (Q8, Q5_0, Q6_K) connect directly to these shared signals. Only Q4_K uses the separate `q4k_wt_*` bundle, which is never driven.

**Impact:** When `matmul_top.v` processes a Q4_K descriptor via the HP burst path, the Q4_K core's `block_buf` is never loaded. The core will compute with all-zero weights, producing garbage results.

**Fix:** Either drive `q4k_wt_we` from the FSM's load logic, or connect `u_core_q4k.wt_we` directly to the shared `wt_we` like the other cores do.

### 2. HP read master ignores `data_ready` (`axihp_read_master.v:94–115`)

In the `DRAIN_BYTE` state, the FSM unconditionally advances to the next byte every cycle regardless of `data_ready`:
```verilog
DRAIN_BYTE: begin
    data_out   <= rdata_buf[buf_idx * 8 +: 8];
    data_valid <= 1;
    // Always advance — FSM is always ready
    if (buf_idx == 7) begin
        ...
    end else begin
        buf_idx <= buf_idx + 1;   // <-- advances whether consumer is ready or not
    end
end
```

**Impact:** If the consumer (`matmul_top.v` FSM) ever de-asserts `hp_read_ready`, the read master will overwrite the byte before it's consumed. Currently works because the consumer is always ready, but fragile.

**Fix:** Gate the advance on `data_ready`.

---

## MODERATE ISSUES

### 3. FP16→fixed-point mismatch between C++ sim and Verilog cores

The C++ simulation (`read_f16` in `fpga_sim.hpp:113`) converts FP16 to full FP32. The Verilog cores use a simplified S24.8 fixed-point approximation:

| Component | Method | Precision |
|-----------|--------|-----------|
| C++ `read_f16` | IEEE 754 FP32 | Full FP32 |
| `matmul_q4k_core.v:126-134` | Custom S24.8 | ~6 decimal digits |
| `matmul_q5_0_core.v:133-135` | Custom fixed-point with rounding | Approximate |
| `matmul_q8_core.v:209-221` | UQ8.8 integer multiply | Fixed-point Q8.8 |

**Impact:** The C++ simulation (which drives the testbenches and `.hdr` golden reference files) uses FP32 arithmetic, while the Verilog cores compute in S24.8 fixed-point. The golden reference values written by `--dump-phaseb` are computed with FP32 precision, so the Verilog Phase B cosimulation (`tb_phaseb.v`) may show mismatches purely due to quantization differences — not actual logic bugs.

**Fix:** Add a C++ simulation mode that matches the Verilog cores' fixed-point arithmetic exactly (e.g., simulate using S24.8 for Q4_K/Q5_0/Q6_K decode).

### 4. Write master `awlen` truncation (`axihp_write_master.v:52`)

```verilog
m_axi_awlen <= word_count[7:0] - 1;
```

If `word_count > 256`, the truncation to 8 bits wraps around, causing incorrect burst length. The caller (`matmul_top.v:872`) computes `hp_write_count <= desc_tile_res_rows` which is at most 64, so this is safe currently. But the module has no assertion.

**Fix:** Add an `if (word_count > 256) ...` guard or an assertion.

### 5. AXI-Lite write FSM requires both AWVALID and WVALID same cycle (`matmul_top.v:303`)

```verilog
if (s_axil_awvalid && s_axil_wvalid) begin
```

The standard AXI4-Lite protocol allows AW and W channels to assert independently with different timing. This design requires both in the same cycle. Real ARM PS7 drivers may separate the address and data phases.

**Impact:** Works in the simulation testbench and Vitis HLS-style drivers (which always send both together), but may fail with standard AXI drivers.

**Fix:** Add a buffer to hold the address until write data arrives.

### 6. `axilite_slave.v` — orphan module

`verilog/axilite_slave.v` is a standalone AXI4-Lite slave module. It is **not instantiated** anywhere in the design. The AXI4-Lite logic is implemented inline in `matmul_top.v` and `axi_wrap_int16.v`.

**Impact:** Dead code. This file could be misleading to someone reading the design.

---

## MINOR ISSUES

### 7. Duplicate comment in Q4_K path (`tmac_gguf.cpp:1104-1105`)

```cpp
// == 896×256 Path: for tensors with cols >= 1024 (down_proj [896, 4864]) ==
    // == 896×256 Path: for tensors with cols >= 1024 (down_proj [896, 4864]) ==
```

Line 1104 and 1105 are identical. The leading `//` alignment is also off.

### 8. Unused function `dequant_q8_to_int16` (`tmac_gguf.cpp:249-278`)

This inline function is defined but never called from any simulation path. It reads `scale_raw` (line 254) then immediately casts to `void` (line 255). It was perhaps intended for the Q8→INT16 conversion path but `matmul_fpga_q8` uses a different method (UQ8.8 combined scales).

### 9. FP16 encoding in testbench has rounding issues (`tb_matmul_q4k.v:72`)

```verilog
mant = $rtoi((abs_val - 1.0) * 1024.0 + 0.5);
```

This `+ 0.5` is a simple round-half-up, not the IEEE-compliant round-to-nearest-even used by `fpga_sim.hpp:write_f16`. This means the testbench-generated Q4_K blocks may differ from what the C++ simulation produces.

### 10. `fpga_sim.hpp:g_timing` is a global mutable struct

```cpp
inline TimingStats g_timing;
```

Used as a global accumulator across all matmul calls. If multiple threads ever use `MatmulAccel`, the timing stats would be racy. Not an issue in current single-threaded usage.

### 11. PhaseB verification tolerance too tight for fixed-point (`tmac_gguf.cpp:837`)

```cpp
const int64_t max_err = 2; // allow ±2 in token output
```

This tolerance is in the dequantized logit space (FP32), not in the accumulator space. If `x_scale` or `row_inv` values are large, a ±2 accumulator error can amplify. For token generation this is fine (small rounding changes don't flip top-1), but the verification uses this to compare against expected values computed via a different path.

### 12. `extract_tmac.py` is a Python script for model conversion but has no associated test

The GGUF→TMAC converter is a critical piece of the toolchain but has no automated test. If the GGUF format changes (e.g., new quantization types), there's no regression detection.

---

## DESIGN NOTES

### Architecture Strengths

1. **Modular core design** — Each quantization format has an isolated core with a uniform interface (`wt_we`/`wt_addr`/`wt_din`, `act_we`/`act_addr`/`act_din`, `res_addr`/`res_dout`). Adding a new quantization type would require only: a new core module, an instantiation in `matmul_top.v`, a mux entry, and a dispatch path in `tmac_gguf.cpp`.

2. **Phase B descriptor chain** — The OP→OP linked list design cleanly separates the CPU from the FPGA workload. The CPU only needs to set up descriptors once; the FPGA runs autonomously.

3. **Cosimulation infrastructure** — The tile dump files (`--dump-tiles`, `--dump-tiles-q5-0`, `--dump-tiles-q6-k`) provide a clean way to feed actual model data into Verilog testbenches.

### Architecture Concerns

1. **No error recovery in descriptor chain** — If a HP burst fails or returns bad data, the chain has no error flag or abort mechanism. The `reg_chain_ctrl` has a reset bit but no way to read back error status per descriptor.

2. **Single-threaded simulation model** — `MatmulAccel` uses a background thread but the DDR copy and register writes happen sequentially in the main thread. This accurately models the ARM-side work but underestimates parallelism.

3. **Weight buffer sizing** — The Q4_K core's `block_buf` is exactly `TILE_ROWS * BLOCK_BYTES = 56*144 = 8064` bytes (`matmul_q4k_core.v:39`). The `write_ptr` auto-increment condition checks `write_ptr < BLOCK_BUF_BYTES` (line 43), but there's no overflow protection at load time if the ARM tries to load more bytes.

4. **No pipeline hazard handling in matmul_top.v** — The FSM does not handle contention between the write and read AXI-Lite FSMs. If the ARM issues a read while a write is in progress, the responses could interleave. The current address set (control registers only, no memory buffers) makes this unlikely but not impossible.

---

## OBSOLETE DOCUMENTATION

The AGENTS.md states matmul_top.v has "3 cores instantiated (Q8, Q4K, INT16)" but the actual file has all 5 cores (Q8, Q4K, INT16, Q5_0, Q6_K). The table in "Existing Verilog Cores" section needs updating.
