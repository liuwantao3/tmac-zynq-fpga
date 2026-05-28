# FPGA Accelerator — Analysis & Task Plan

> Status: May 2026 — Post-iVerilog verification, pre-Vivado bitstream.
> All core RTL passes simulation and end-to-end inference.

---

## 1. Problem: The Timing Model is Wrong

### 1.1 What the Simulation Reports

```
[FPGA TIMING SUMMARY]
  Tiles: 78740     MACs: 493961216
  FPGA: 17289580 cycles @ 150 MHz = 115263.87 us  Speedup: 11.4x
```

**Claimed: 8.7 tokens/sec, 11.4× over CPU (1319 ms naive).**

### 1.2 Why It's Wrong

The cycle counter only counts the **INT16 fallback** path (`MatmulAccel`):

```cpp
// sim/fpga_sim.hpp:313
total_cycles_ += num_tiles * CYCLES_PER_TILE;  // 515 cy/tile, INT16 only
```

The quantization-specific paths (Q5_0, Q6_K, Q4_K, Q8_0) increment `total_tiles` but add **zero** to `total_fpga_cycles`. These paths account for **75% of tiles** (59K of 79K).

The timing report at `tmac_gguf.cpp:2169`:
```cpp
fpga_sim::g_timing.total_fpga_cycles += fpga_sim::accel().total_cycles();
```

Uses `accel().total_cycles()` which is the MatmulAccel INT16 count only. The Q-path cycles are invisible.

**Result**: reported "FPGA time" is 115 ms for INT16 tiles only. Real time including Q-path compute + DDR loading would be **much higher** (see §3).

### 1.3 Root Cause — No DDR or FSM Timing in the Model

The simulation (`fpga_sim.hpp`) was originally written for the simple `MatmulAccel` HLS model where weights were pre-loaded into a local DDR buffer at simulation startup. When the Q-paths were added (bypassing MatmulAccel with custom `axi_vecmul_tile_*` functions), nobody updated the timing model to account for:
- Actual per-core compute cycles (14,337 for Q5_0, not 515)
- DDR weight loading time via AXI HP
- Activation copy overhead
- Result write-back overhead
- FSM sequential phases (no overlap)

---

## 2. Current Architecture — Three Inefficiencies

### 2.1 HP Read Master: 9 Cycles per AXI Beat

`verilog/axihp_read_master.v` unpacks each 64-bit AXI beat into 8 bytes serially:

```
WAIT_R: capture beat → m_axi_rready <= 0    // 1 cycle, stalls R channel
DRAIN_BYTE: output 1 byte/cycle × 8          // 8 cycles
                                              // = 9 cycles/beat
```

Since the R channel is stalled during drain, DDR throughput is **8× worse** than necessary.

**Impact per tile:**

| Tile Type | Bytes | AXI Beats | Cycles (current) | Cycles (64-bit ideal) |
|-----------|-------|-----------|-----------------|----------------------|
| Q5_0 (8×896) | 4,928 | 616 | **5,544** | 616 |
| Q6_K (32×256) | 6,720 | 840 | **7,560** | 840 |
| Q4_K (56×256) | 8,064 | 1,008 | **9,072** | 1,008 |
| Q8_0 (64×64) | 4,096 | 512 | **4,608** | 512 |

**Total DDR load cycles per token (current):** ~460M cycles  
**Total DDR load cycles per token (64-bit fix):** ~52M cycles — **8.8× improvement**

### 2.2 Activation Copy Every Tile

`matmul_top.v` FSM does `PH_WRITE_ACT` **inside the per-tile loop**. The activation vector is the same for all tiles in a descriptor (same input column). This copies `act_buf → core` redundantly.

For a 112-tile Q5_0 descriptor (attn_q): **111 of 112 copies are wasted**.
Total waste: 111 × 896 cycles = **99,456 cycles** per descriptor.

**Fix**: move `PH_WRITE_ACT` to once after `PH_LOAD_ACT`, before entering tile loop.

### 2.3 Sequential FSM — No Load/Compute Overlap

```
Per tile: LOAD_WEIGHT → WRITE_ACT → COMPUTE → WRITE_RESULT
                        (all sequential, no overlap)
```

Compute cannot start until all weights are loaded. The next tile's weights cannot load until results are written back.

**Fix**: double-buffered weight buffers. Tile N computes from buf_A while tile N+1 loads into buf_B. Requires:
- Two weight buffers per core (2× BRAM)
- FSM: immediately trigger next tile weight load after starting compute
- Result write can also overlap with compute in principle

---

## 3. Real Performance Analysis

### 3.1 Actual Per-Core Cycle Counts (from RTL)

| Core | Tile | MACs/tile | Compute cycles | MACs/cycle | Notes |
|------|------|-----------|---------------|------------|-------|
| Q5_0 | 8×896 | 7,168 | **14,337** | 0.5 | 2-cycle LD+DEC per element, no pipeline |
| Q6_K | 32×256 | 8,192 | **8,193** | ~1.0 | 1-cycle, decode combinational |
| Q4_K | 56×256 | 14,336 | **14,337** | ~1.0 | 1-cycle, decode combinational |
| Q8_0 | 64×64 | 4,096 | **514** | ~8.0 | 3-stage pipeline, 8-wide SIMD |
| INT16 | 64×64 | 4,096 | **514** | ~8.0 | Same pipeline as Q8_0 |

### 3.2 Tile Count Breakdown

Total tiles per token (from simulation with `--fpga-q5-0 --fpga-q6-k --fpga-q4k --fpga-q8`): **78,740**

| Source | Tensor Shape | Tile Type | Tiles/token | % of total |
|--------|-------------|-----------|-------------|-----------|
| attn_q × 28 layers | 896×896 | Q5_0 | 3,136 | 4.0% |
| attn_k × 28 layers | 896×896 | Q5_0 | 3,136 | 4.0% |
| attn_output × 28 layers | 896×896 | Q5_0 | 3,136 | 4.0% |
| ffn_gate × 28 layers | 4864×896 | Q5_0 | 17,024 | 21.6% |
| ffn_up × 28 layers | 4864×896 | Q5_0 | 17,024 | 21.6% |
| ffn_down even × 14 layers | 896×4864 | Q6_K | 7,448 | 9.5% |
| ffn_down odd × 14 layers | 896×4864 | Q4_K | 4,256 | 5.4% |
| attn_v × 28 layers | 128×896 | Q8_0 | 784 | 1.0% |
| INT16 fallback (norms, etc.) | various | INT16 | ~22,796 | 28.9% |

### 3.3 Current Real Performance Estimate

Using actual per-tile cycle costs (including byte-serial weight load, act copy, compute, result write):

| Component | Tiles | Cycles/tile | Total cycles | Time @150 MHz |
|-----------|-------|-------------|-------------|---------------|
| Q5_0 compute | 43,456 | 14,337 | 623M | 4,153 ms |
| Q5_0 weight load (byte-serial) | 43,456 | 5,544 | 241M | 1,607 ms |
| Q5_0 act copy | 43,456 | 896 | 39M | 260 ms |
| | | | **Q5_0 subtotal** | **6,020 ms** |
| Q6_K compute | 7,448 | 8,193 | 61M | 407 ms |
| Q6_K weight load | 7,448 | 7,560 | 56M | 373 ms |
| Q6_K act copy | 7,448 | 256 | 2M | 13 ms |
| | | | **Q6_K subtotal** | **793 ms** |
| Q4_K compute | 4,256 | 14,337 | 61M | 407 ms |
| Q4_K weight load | 4,256 | 9,072 | 39M | 260 ms |
| Q4_K act copy | 4,256 | 256 | 1M | 7 ms |
| | | | **Q4_K subtotal** | **674 ms** |
| Q8_0 compute | 784 | 514 | 0.4M | 3 ms |
| Q8_0 weight load | 784 | 4,608 | 3.6M | 24 ms |
| Q8_0 act copy | 784 | 64 | 0.05M | 0.3 ms |
| | | | **Q8_0 subtotal** | **27 ms** |
| INT16 compute | 22,796 | 514 | 12M | 80 ms |
| INT16 weight load | 22,796 | 4,608* | 105M | 700 ms |
| | | | **INT16 subtotal** | **780 ms** |
| **Total** | **78,740** | | | **~8,300 ms** |

*\*INT16 = same byte-serial as Q8_0*

**Current real performance: 0.12 tokens/sec** (vs claimed 8.7 tokens/sec).  
CPU naive (single-thread): **0.76 tokens/sec** (1319 ms).  
**FPGA is 6.3× slower than CPU.**

### 3.4 Why This Happened

The accelerator was designed and verified without a realistic DDR bandwidth model. The C++ simulation's `axi_vecmul_tile_*` functions skip DDR loading entirely (data is already in local buffers). The timing counter is a placeholder (515 cycles/tile) that was never updated for:
- The actual core compute cycles (discovered during iVerilog testbench work)
- The DDR loading overhead (not modeled until the byte-drain analysis above)

---

## 4. The DSP Opportunity

### 4.1 Current DSP Usage

Zynq 7010 has **80 DSP48E1 slices**. Current usage:

| Core | DSPs used | MACs/cycle | Bottleneck |
|------|-----------|-----------|-----------|
| Q5_0 | 1 (LUT for dequant) | 0.5 | Sequential 2-cycle LD+DEC |
| Q6_K | 1 (~LUT for dequant) | ~1.0 | Sequential, 8,193 cycles |
| Q4_K | 1 (~LUT for dequant) | ~1.0 | Sequential, 14,337 cycles |
| Q8_0 | 8 (8-wide SIMD) | ~8.0 | |
| INT16 | 8 (8-wide SIMD) | ~8.0 | |
| **Peak** | **8** | | **72 DSPs idle** |

### 4.2 Why DSPs Are Underused

The Q5_0/6_K/4_K cores were designed for **LUT-based dequant + 1 DSP for MAC**, mimicking a microcontroller approach (one multiply at a time). The dequant math exceeds DSP48E1's 25×18 multiplier (FP32 × int5 involves 32-bit multiplies for the FP16→FP32 conversion), so it's done in LUTs. But the final INT16×INT16 MAC is trivially DSP48E1-compatible.

### 4.3 Proposed: Shared DSP MAC Pool

Since only one core runs at a time (mutual exclusion via `mode_*` in `reg_ctrl_user`), DSPs can be **shared** across a common MAC pool:

```
          ┌─────────────┐
Weight ──►│ Block Decode│──► val[row][col] ──┐
          │(LUT, 1 cyc) │                     │
          └─────────────┘                     ▼
                                        ┌──────────┐
Act ───────────────────────────────────►│ DSP Pool │──► accum[row]
                                        │ (64 ×    │
                                        │  DSP48E1)│
                                        └──────────┘
```

The MAC pool does: `accum[row] += val[row] × act[col]` in one cycle per DSP.

### 4.4 Per-Core DSP Allocation

| Core | Tile | MACs | DSPs | Strategy | Est. Compute Cycles |
|------|------|------|------|----------|-------------------|
| Q5_0 | 8×896 | 7,168 | **64** (8 rows × 8 cols/cyc) | 32 col-groups × 5 cy = 140 cy/tile | **41 ms** |
| Q6_K | 32×256 | 8,192 | **32** (1 per row) | 256 cols / 8 col/cy = 32 cy × 32 row-grps | **~400 ms** |
| Q4_K | 56×256 | 14,336 | **56** (1 per row) | 256 cols / 4 col/cy = 64 cy × 16 row-grps | **~270 ms** |
| Q8_0 | 64×64 | 4,096 | **8** (already done) | Current pipeline works | **27 ms** |

**Peak DSP demand: 64** (Q5_0 path) — well within 80 available.  
Q6_K needs 32, Q4_K needs 56 — all under the 80 limit.

### 4.5 Q5_0 64-DSP Design Detail

**Data flow for one 8×32 column-group:**

1. **Cycle 0**: Read 8 block headers (d + qh) from 8 parallel row-BRAMs
2. **Cycles 1-4**: For each of 4 column-groups × 8 columns:
   - 8 lane-decoders × 8 rows = 64 parallel decoders (pure combinational)
   - Each decoder: extract ql nibble + qh bit, compute `q5_val = ((qh_bit<<4)|ql)-16`
   - 64 DSPs: `acc[row] += d_fp(row) × q5_val(row, col) × act[col]`
   - One cycle per column-octet → 4 cycles
3. **Total: 5 cycles per column-group × 28 groups = 140 cycles per tile**

**Block-buf organization** (8 BRAMs, one per row):
```
BRAM[row]: holds 28 blocks × 22 bytes = 616 bytes
          → reorganized as 77 words × 64 bits (3 BRAM reads per block)
          → cycle 0 reads d(2B) + qh(4B) + first nibbles(2B) = 8B
          → remaining nibble data streamed over the next 32 cycles
```

Actually, a cleaner approach: pre-decode the block header in cycle 0 into 32× 5-bit values stored in 4 × 32-bit registers. Then cycles 1-4 read 8 values from these registers per lane per cycle. This avoids the nibble extraction latency during the hot MAC loop.

### 4.6 Impact of 64-DSP Redesign

| Fix Level | Q5_0 compute | All compute | DDR load (64-bit HP) | Total | Tokens/sec |
|-----------|-------------|-------------|-------------------|-------|-----------|
| **Current (no fixes)** | 6,020 ms | 8,300 ms | incl. in load | ~8,300 ms | **0.12** |
| + 64-bit HP read (Fix 1) | 4,153 ms | 5,646 ms | ~300 ms | ~5,946 ms | 0.17 |
| + shared act (Fix 2) | 3,450 ms | 4,900 ms | ~300 ms | ~5,200 ms | 0.19 |
| + double-buffer (Fix 3) | 2,070 ms | 3,000 ms | ~150 ms | ~3,150 ms | 0.32 |
| **+ 64-DSP Q5_0** | **41 ms** | **1,530 ms** | ~300 ms | **~1,830 ms** | **0.55** |
| + 64-DSP all cores | 41 ms | **~90 ms** | ~300 ms | **~390 ms** | **2.56** |
| + ideal DDR (2.1 GB/s) | 41 ms | ~90 ms | ~186 ms | **~277 ms** | **3.61** |

**With all DSP fixes + DDR optimizations, we approach the practical ceiling of 3-4 tokens/sec set by the DDR3 controller bandwidth (2.1 GB/s ÷ 390 MB model = 5.4 tokens/sec theoretical).**

---

## 5. Comparison: FPGA vs CPU

### 5.1 Throughput

| Implementation | Tokens/sec | Relative |
|---------------|-----------|----------|
| CPU naive (Cortex-A9, 1 thread, no SIMD) | 0.76 | 1.0× |
| CPU optimized (NEON SIMD, estimate) | ~3.0 | ~4× |
| FPGA current (no fixes) | 0.12 | 0.16× (6× slower) |
| FPGA + 64-DSP + DDR fixes | 2.56 | 3.4× |
| FPGA + all optimizations | 3.61 | 4.7× |

### 5.2 Power

- Cortex-A9 at 667 MHz: ~3W (with L2 cache + SCU)  
- Zynq 7010 PL at 150 MHz: ~1.5W (typical for this utilization)

The FPGA is competitive on **power efficiency**: at 3.6 tokens/sec and 1.5W = 2.4 tokens/sec/W vs CPU at 0.76 tokens/sec and 3W = 0.25 tokens/sec/W. **FPGA is ~10× more power-efficient.**

### 5.3 Strategic Fit

The FPGA accelerator makes sense for:
- **Battery-powered** or **thermal-constrained** deployments (the original Xilinx use case)
- **Deterministic latency** applications (real-time control)
- **CPU offload** — the Cortex-A9 runs the host application while FPGA does inference

It does **not** make sense for:
- Raw throughput comparison vs modern CPUs/GPUs
- Replacing a multi-core application processor

---

## 6. Task List

### P0 — Fix the Timing Model (sim-only, no RTL changes)
- [ ] Add per-core cycle constants to `fpga_sim.hpp`: `Q5_0_CYCLES=14337`, `Q6_K_CYCLES=8193`, `Q4_K_CYCLES=14337`, `Q8_0_CYCLES=514`
- [ ] Add DDR load cycle model: `weight_bytes / 8 * 9` for current byte-serial, `weight_bytes / 8` for 64-bit mode
- [ ] Add FSM overhead model: act copy, result write, per-tile iteration
- [ ] Update `g_timing.report()` to use actual per-type cycles instead of `total_tiles * N`
- [ ] Add `--accurate-timing` flag to toggle between old (fast) and new (accurate) models
- [ ] Update AGENTS.md with corrected performance numbers

### P0 — Fix 1: 64-bit HP Read Master
- [ ] New module `axihp_read_master_64.v`: output `data_out[63:0]`, `data_valid`, `data_ready` — one 64-bit word per beat, no byte-serial drain
- [ ] Update `matmul_top.v` instantiation
- [ ] Widen all 5 core wt_din ports: 8-bit → 64-bit
- [ ] Change block_buf in Q4_K/Q5_0/Q6_K from byte-addressed `reg [7:0]` to word-addressed `reg [63:0]`
- [ ] Update wmem in Q8_0/INT16 to accept 64-bit writes directly
- [ ] Update FSM: `PH_WEIGHT_WAIT` writes one 64-bit word per cycle instead of 8 bytes
- [ ] Verify: run existing testbenches, compare tile dumps

### P1 — Fix 2: Shared Activation per Descriptor
- [ ] Move `PH_WRITE_ACT` from inside per-tile loop to after `PH_LOAD_ACT_WAIT`
- [ ] Verify activation buffer retains values across core re-starts
- [ ] Verify: all testbenches pass with tile-by-tile comparison

### P1 — Fix 3: Double-Buffered Weight Load
- [ ] Add top-level `weight_buf_A` and `weight_buf_B` (each 8192 bytes, BRAM, one additional per core)
- [ ] FSM change: during compute of tile N, trigger weight load of tile N+1 into the opposite buffer
- [ ] Core interface: add `buf_sel` input (0/1), mux weight reads between A/B
- [ ] Handle first tile (no preloaded buffer) and last tile (no next load)
- [ ] Verify with testbenches

### P2 — 64-DSP Shared MAC Pool
- [ ] Design DSP pool module: `dsp_mac_pool.v`
  - `num_dsp` parameter (8/32/56/64)
  - Input: `val[63:0][15:0]`, `act[15:0]`, `accum[63:0][47:0]`
  - Output: `accum_next[63:0][47:0]`
  - Instantiates `num_dsp` DSP48E1 primitives in `A×B+C` mode
- [ ] Q5_0 64-DSP decoder redesign:
  - 8 parallel block_buf BRAMs (one per row)
  - Pre-decode block header (d + qh) to 32× 5-bit values per row in 1 cycle
  - Read 8 columns × 8 rows = 64 values per cycle, feed to DSP pool
  - 4 cycles per column-octet, 5 cycles per column-group, 140 cycles per tile
- [ ] Q6_K 32-DSP redesign (similar pattern)
- [ ] Q4_K 56-DSP redesign (similar pattern)  
- [ ] Mux DSP pool between cores (only one active at a time)
- [ ] Verify with cosimulation (tile-by-tile comparison against C++ reference)

### P3 — Model & Documentation Updates
- [ ] Update `fpga_caps.json` with realistic bandwidth/compute ceilings
- [ ] Update `AGENTS.md` with corrected performance and analysis
- [ ] Document DDR bandwidth vs compute trade-off in `docs/architecture.md`

---

## 7. Key Equations

### Cycle Count per Tile (General Form)

```
tile_cycles = weight_load(tile_bytes) + act_copy(cols) + compute(macs) + result_write(rows)
```

Where:
- `weight_load(B) = ceil(B / 8) × beat_penalty` — beat_penalty = 9 (current) or 1-2 (64-bit fix)
- `act_copy(C) = C` — one cycle per activation value (can skip after first tile in descriptor)
- `compute(M) = core_cycles(macs)` — depends on core pipeline
- `result_write(R) = ceil(R × 8 / B_w)` — B_w = write bus width

### DDR Bandwidth Limit

```
max_tokens_per_sec = DDR_BW / model_weight_bytes
                   = 2.13 GB/s / 390 MB
                   ≈ 5.5 tokens/sec (theoretical)
```

DDR3 on Zynq 7010: 16-bit @ 533 MHz (1066 MT/s) = 2.13 GB/s peak through a single memory controller. The 4.7 GB/s figure often quoted is 4× HP ports × 1.2 GB/s each, ignoring the DDR controller bottleneck.

### MACs vs DSPs

```
DSPs_needed = ceil(MACs_per_cycle / 1)
```

Each DSP48E1 can do one 18×18+48 MAC per cycle (`A × B + C = P`). For a tile with M MACs and P DSPs:

```
compute_cycles = ceil(MACs / P) 
               + decode_overhead(blocks, rows)
```

Where `decode_overhead` is the cycles needed before MAC operations begin (loading block headers, setting up decoders).

---

## 8. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| 64-bit weight writes don't fit BRAM port width | Block-buf reorganizations needed, more BRAMs | Use dual-port BRAM; one port for load, one for compute |
| 64 parallel decoders exceed LUT budget | ~200 LUTs per decoder × 64 = 12,800 LUTs; 7010 has 44,000 | Acceptable at 29% of LUTs |
| DSP pool mux adds timing closure risk | Extra LUTs + routing on DSP cascade path | Pipeline muxed inputs; one cycle penalty for switching cores |
| DDR controller contention with PS7 | Reduced effective bandwidth | Use dedicated HP port; minimize CPU DDR access during inference |
| Vivado 2019 toolchain issues | Synthesis failures, P&R timeouts | Use proven `clean -purge` workflow; incremental builds |

---

## 9. Quick Reference — Per-Tile Cycle Budgets

### Current (byte-serial, no fixes)

| Component | Q5_0 (8×896) | Q6_K (32×256) | Q4_K (56×256) | Q8_0 (64×64) |
|-----------|-------------|--------------|--------------|-------------|
| Weight load | 5,544 | 7,560 | 9,072 | 4,608 |
| Act copy | 896 | 256 | 256 | 64 |
| Compute | 14,337 | 8,193 | 14,337 | 514 |
| Result write | ~10 | ~30 | ~50 | ~60 |
| **Total** | **20,787** | **16,039** | **23,715** | **5,246** |

### After Fix 1 (64-bit HP read) + Fix 2 (shared act)

| Component | Q5_0 | Q6_K | Q4_K | Q8_0 |
|-----------|------|------|------|------|
| Weight load | 616 | 840 | 1,008 | 512 |
| Act copy | 896* | 256* | 256* | 64* |
| Compute | 14,337 | 8,193 | 14,337 | 514 |
| Result write | ~10 | ~30 | ~50 | ~60 |
| **Total** | **15,859** | **9,319** | **15,651** | **1,150** |

*\*Once per descriptor, not per tile*

### After Fix 3 (double-buffer overlap)

Effective cycles per tile (amortized): max(weight_load + act_copy_one, compute + result_write)  
For Q5_0: max(616 + 0, 14,337 + 10) = **14,347**

### After 64-DSP Q5_0

| Component | Cycles | Notes |
|-----------|--------|-------|
| Block header load | 28 | 1 cycle per column-group |
| Compute (64 MACs/cycle) | 112 | 28 groups × 4 cycles |
| **Total** | **140** | 51.2 MACs/cycle avg |
