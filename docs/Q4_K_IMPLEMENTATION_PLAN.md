# Q4_K FPGA Accelerator — Implementation Plan

> **STATUS: IMPLEMENTED (2026-07-05).** The actual implementation is in
> `verilog/matmul_q4k_core.v` with a 56×256 tile and block-streaming decode.
> This document represents the planning phase and differs from the final
> implementation in several key areas (tile size, memory architecture, etc.).

## 1. Overview

### Current State
- **Accelerator**: Q8_0 only (`matmul_q8_core.v` + `matmul_q8_top.v`)
- **Tile**: 64×64, 3-stage pipeline, 515 cycles/tile
- **Target Model**: Qwen 2 0.5B (`qwen2-0_5b-instruct-q4_k_m.gguf`)
- **Model Dimensions**: H=896, INTER=4864, 24 layers

### Why Q4_K
| Metric | Q8_0 | Q4_K | Improvement |
|--------|------|------|-------------|
| Weight bits | 8 | 4 | 2× smaller |
| Tile data (64×64) | 4096 B | 2304 B | 44% less |
| DDR bandwidth | baseline | **2× savings** | LLMs are bandwidth-bound |

### Target Topology (Qwen 2 0.5B q4_k_m)

In `q4_k_m`:
- **FFN gate/up/down** → Q4_K (largest, ~70% of weights)
- **Attention Q/K/V/O** → Q6_K or Q4_K (depending on variant)
- **Token embeddings** → typically F16

**Primary target = FFN** (gate + up + down = 3 × 896 × 4864 = ~13M weights per layer × 24 layers = ~312M weights total = largest workload)

---

## 2. Q4_K Format Analysis

### Q4_K Block Structure (256 weights = 144 bytes)

```
Byte offset  Size  Field      Description
──────────── ────  ─────────  ─────────────────────────────────
0-1           2B   d          Block delta (float16)
2-3           2B   dmin       Block minimum (float16)
4-15         12B   scales     Packed 6-bit scales (8 × sc + 4 × m)
16-143      128B   qs         256 × 4-bit weight values (2 per byte)
                           ─────────
Total       144B              256 weights → 0.5625 B/weight
```

### Sub-block Structure (per 32 weights)

| sub_block | offset range | sc source | m source | q4 extraction |
|-----------|-------------|-----------|----------|---------------|
| 0 | 0-31 | scales[0] & 0x3F | scales[4] & 0x3F | qs[n] & 0xF |
| 1 | 32-63 | scales[1] & 0x3F | scales[5] & 0x3F | qs[n] >> 4 |
| 2 | 64-95 | scales[2] & 0x3F | scales[6] & 0x3F | qs[n] & 0xF |
| 3 | 96-127 | scales[3] & 0x3F | scales[7] & 0x3F | qs[n] >> 4 |
| 4 | 128-159 | (sc[8] & 0xF) \| (sc[0] >> 6)<<4 | (sc[8] >> 4) \| (sc[4] >> 6)<<4 | qs[n] & 0xF |
| 5 | 160-191 | (sc[9] & 0xF) \| (sc[1] >> 6)<<4 | (sc[9] >> 4) \| (sc[5] >> 6)<<4 | qs[n] >> 4 |
| 6 | 192-223 | (sc[10] & 0xF) \| (sc[2] >> 6)<<4 | (sc[10] >> 4) \| (sc[6] >> 6)<<4 | qs[n] & 0xF |
| 7 | 224-255 | (sc[11] & 0xF) \| (sc[3] >> 6)<<4 | (sc[11] >> 4) \| (sc[7] >> 6)<<4 | qs[n] >> 4 |

### Dequantization Formula
```
dequant_val = d * (sc * q4) - dmin * m

Where:
  d    = block delta (float16, 1 per 256 weights)
  dmin = block minimum (float16, 1 per 256 weights)
  sc   = sub-block scale (6-bit, 0-63, 8 per block)
  m    = sub-block min mul (6-bit, 0-63, 8 per block)
  q4   = 4-bit weight (0-15)
```

---

## 3. Architecture Design Decision

### Approach Selected: Pre-Dequantize Q4_K → INT16 During Loading

```
C++ writes Q4_K bytes ──AXI──▶ weight_buf (top)
                                      │
                          loading FSM converts Q4_K → INT16
                                      │
                                  wmem (core, now INT16)
                                      │
                          Pipeline: act[k] × wmem[{g,k}] → acc
```

### Rationale

| Approach | Complexity | BRAM | Pipeline Changes | CPU Load |
|----------|-----------|------|-----------------|---------|
| **A: Pre-dequant in loading** | Medium | 2× BRAM | None (INT16 MAC) | Same |
| B: Dequant in compute pipeline | High | Same | Major (4-bit pack, scale LUT) | Same |
| C: CPU dequant, send INT16 | Low | 2× BRAM | None | **More** |
| D: CPU dequant, send Q8_0 | Lowest | Same | None | Most (lossy) |

**Winner: A** — moves dequantization work from CPU to FPGA, keeps pipeline simple, moderate HW change.

### Key Design Changes

| Component | Q8_0 | Q4_K | Change |
|-----------|------|------|--------|
| `weight_buf` | 4096 × 8-bit | 4096 × 8-bit | Same (stores Q4_K raw bytes) |
| `wmem` | 512 × 64-bit (BRAM) | 1024 × 32-bit (2 BRAMs) | **2× depth, ½ width** |
| `smem` | 128 × 16-bit | Removed | Scales handled in wmem |
| Loading FSM | Byte copy wbuf→wmem | Q4_K→INT16 dequant | **Major change** |
| Pipeline | INT8×UQ8.8 MAC | INT16×INT16 MAC | Minor (different types) |
| Dequant func | `dequant_q8` | None (pre-dequant) | Removed from pipeline |

---

## 4. Detailed Hardware Design

### 4.1 New wmem Layout

```
Q8_0 (current):   wmem[0:511] × 64-bit = 1 BRAM
  addr = {g, k} → 8 bytes = 8 Q8 weights

Q4_K (new):       wmem[0:1023] × 32-bit = 2 BRAMs
  addr = (g*8 + wi_i)*2*64 + k  → 32-bit = 2 INT16 values
  But we want 8 INT16 per iteration (one per row in bank group)

  Better: wmem[0:511] × 64-bit = 2 BRAMs (dual-port, 64-bit each)
  addr = {g, k} → 2 × 32-bit = 4 INT16 values
  Need 2 reads per bank group
```

Actually, we need to carefully think about this:

For 64×64 tile:
- 64 rows × 64 columns = 4096 INT16 values
- Each INT16 = 2 bytes
- Total: 8192 bytes

Available BRAMs: can use 2 BRAMs
- 2 × 512 × 72-bit = 1024 × 72-bit organized as 1024 × 64-bit

Best layout: **wmem[0:511] × 128-bit** = 2 BRAMs in parallel (128-bit = 8 INT16 values)
```
wmem[{g, k}] = 128-bit = 8 INT16 values (one per row in bank group)
```

This perfectly matches the 8-lane pipeline. Same iteration pattern as Q8_0!

### 4.2 Loading FSM — Q4_K Dequantization Phase

The loading stage needs a new phase between LP_WEIGHT and LP_SCALE:

```
LP_WEIGHT   → copy Q4_K raw bytes from weight_buf to local Q4 block buffer
LP_Q4_DEQ   → dequantize 16 blocks × 256 weights → 4096 INT16 → write to wmem
LP_SCALE    → skip (scales already accounted)   OR keep for activations
LP_ACT      → copy act_buf as before
LP_DONE
```

#### Q4_K Dequantization Logic for FPGA

For each block (16 blocks per tile, 4 rows each):

```
Input: Q4_K block data (144 bytes from weight_buf)
Output: 256 INT16 values → wmem

Processing flow per sub-block (×8 per block, 32 weights each):
  1. Read sc, m from scales array (6-bit extraction)
  2. For each of 32 weights:
     a. Extract 4-bit value from qs array (nibble)
     b. Compute: int16 = saturate_i16(d * (sc * q4) - dmin * m)
     c. Write to wmem at appropriate address
```

The key challenge: float16 → fixed-point computation for d and dmin.

#### Float16 → UQ8.16 Conversion

To avoid floating-point hardware:

```verilog
// d_man = d mantissa (10-bit), d_exp = d exponent (5-bit)
// d = (-1)^sign × (1 + mant/1024) × 2^(exp-15)
// Convert to fixed-point: d_fixed = d * 2^16
// Scale factors sc, q4, m are small integers (0-63, 0-15)

// Compute: d * sc * q4, then shift by exponent
// d_min * m, then subtract

// Approach: precompute d_as_uq8_16 = floor(d * 65536 + 0.5)
// Then: val = (d_as_uq8_16 * sc * q4) >> 16 - (dmin_as_uq8_16 * m) >> 16
```

This is feasible with LUT-based multiply (8-bit × 16-bit = 24-bit).

For each sub-block (32 × 8-bit multiplies), but done during loading (before compute):
- Not timing-critical
- Can use LUT-based sequential multiplier
- Total: 16 blocks × 8 sub-blocks × 32 weights = 4096 dequant operations during loading

At 1 dequant per 2 cycles (sharing 1 multiplier): ~8192 cycles extra loading time.

### 4.3 Pipeline Changes (Core)

The 3-stage pipeline stays almost the same:

```
Q8_0 Pipeline:
  Stage 0:  wmem addr {g,k}, read act[k], read smem[...]
  Stage 1:  dequant_q8(wmem_rdata[wi_i*8 +:8], p1_sc[wi_i]) × act
  Stage 2:  acc[p2_row_base + wi_i] += p2_partial[wi_i]

Q4_K Pipeline:
  Stage 0:  wmem addr {g,k}, read act[k]  (no smem read)
  Stage 1:  p1_partial = wmem_rdata[wi_i*16 +:16] × p1_act  (INT16×INT16)
  Stage 2:  acc[p2_row_base + wi_i] += p2_partial[wi_i]
```

Key simplification: no dequant in pipeline (pre-dequant on loading).

Key change: INT16 × INT16 multiply (uses same 8 DSPs, different operand widths).

### 4.4 Dual-Port BRAM for wmem

Two BRAMs configured as true dual-port:
```
BRAM0: wmem[0:511][63:0]   (lower 64-bit of 128-bit entry)
BRAM1: wmem[0:511][63:0]   (upper 64-bit of 128-bit entry)

Port A: compute pipeline read  (pipeline stage 0)
Port B: loading FSM write      (dequant loading)
```

No contention: loading completes before compute starts.

### 4.5 Mode Switching (Q8_0 / Q4_K)

A new register bit in `REG_CTRL_USER`:

```verilog
localparam CTRL_MODE_Q4K = 1 << 6;  // New bit in ctrl_user
```

When set:
- Loading FSM: uses `LP_Q4_DEQ` phase (in addition to LP_WEIGHT for raw Q4_K bytes)
- wmem addressed same way (same iteration pattern)
- Pipeline: bypass `dequant_q8`, use INT16×INT16 instead of INT8×UQ8.8
- smem reads disabled

When clear (Q8_0 mode):
- Original behavior preserved

---

## 5. Memory Map for Q4_K Mode

### AXI4-Lite Address Map (shared with Q8_0)

| Address Range | Q8_0 Mode | Q4_K Mode |
|--------------|-----------|-----------|
| 0x0000-0x000F | Control registers | Same |
| 0x1000-0x1FFF | weight_buf (4096 bytes = Q8 raw) | weight_buf (2304 bytes Q4_K raw + 256B scales) |
| 0x2000-0x20FF | scale_buf (128 × u16) | Unused (scales in Q4_K block) |
| 0x2100-0x217F | act_buf (64 × i16) | Same |
| 0x4000-0x40FF | result lower 32 bits | Same |
| 0x4200-0x427F | result upper 16 bits | Same |

### Q4_K Tile Format (C++ → weight_buf)

```
Weight buffer layout for Q4_K tile:
  2304 bytes: 16 Q4_K blocks × 144 bytes
              (64×64 weights, col-major, 4 rows/block)

Scale information is WITHIN each block (bytes 0-15 of each 144B block).
No separate scale buffer needed.
```

---

## 6. Resource Budget

### Current (Q8_0)

| Resource | Used | Available | % |
|----------|------|-----------|---|
| DSP48E1 | 8 | 80 | 10% |
| BRAM36 | 1 | 60 | 2% |
| LUT | 3K | 17,600 | 17% |
| FF | 8K | 35,200 | 23% |

### Q4_K Addition

| Resource | Delta | New Total | New % |
|----------|-------|-----------|-------|
| BRAM36 | +1 | 2 | 3% |
| DSP48E1 | 0 | 8 | 10% |
| LUT | +800 | 3.8K | 22% |
| FF | +300 | 8.3K | 24% |

**Breakdown of extra LUTs:**
- Q4_K block parser/scale unpacker: +200 LUTs
- Q4→INT16 dequant (LUT multiplier): +300 LUTs
- Loading FSM control MUX: +100 LUTs
- Mode mux (Q8/Q4): +200 LUTs

### BRAM Usage Detail

```
BRAM0 (Q8_0 mode):  wmem[0:511] × 64-bit  = 1 BRAM36 (single-port)
BRAM0+1 (Q4_K mode): wmem[0:511] × 128-bit = 2 BRAM36 (dual 64-bit)
```

Both BRAMs share address: `{g, k}` — same iteration pattern.

---

## 7. C++ Integration

### 7.1 Software Flow Change

Current (`matmul_fpga_q8`):
```cpp
1. Extract Q8_0 bytes from tensor (tile by tile)
2. Compute combined scales (UQ8.8)
3. Send tile: q8_bytes + combined_scales + vec
4. FPGA: Q8→INT16 dequant + matmul
5. Read results
```

New (`matmul_fpga_q4k`):
```cpp
1. Extract Q4_K block data from tensor (tile by tile)
2. Send tile: q4k_blocks + vec  
3. FPGA: Q4_K→INT16 dequant + matmul
4. Read results
```

### 7.2 New C++ Function

```cpp
void matmul_fpga_q4k(const Tensor* A, const float* x, float* y, int rows, int cols) {
    for (int r = 0; r < rows; r += 64) {
        int r_size = std::min(64, rows - r);
        for (int c = 0; c < cols; c += 64) {
            int c_size = std::min(64, cols - c);
            // Extract 64×64 tile as Q4_K blocks (16 blocks)
            uint8_t q4k_tile[2304];  // 16 × 144 bytes
            extract_q4k_tile(A, r, c, q4k_tile);

            // Quantize activation to INT16
            int16_t vec[64];
            quantize_act(x + c, vec, 64);

            // Set Q4K mode
            accel().write_reg(REG_CTRL_USER, CTRL_OP_VECMUL | CTRL_MODE_INT16 | CTRL_MODE_Q4K);
            // Send Q4_K blocks
            memcpy(accel().ddr() + WEIGHT_OFF, q4k_tile, 2304);
            memcpy(accel().ddr() + ACT_OFF, vec, 128);
            // Start and wait
            accel().write_reg(REG_AP_CTRL, AP_START);
            accel().wait_done();
            // Read results
            int64_t result[64];
            memcpy(result, accel().ddr() + RES_OFF, 64 * 8);
            dequantize_row(result, y + r, r_size);
        }
    }
}
```

### 7.3 Tile Extraction for Q4_K

```cpp
void extract_q4k_tile(const Tensor* A, int row_start, int col_start,
                      uint8_t* q4k_tile_out) {
    // A Q4_K block encompasses 4 rows × 64 columns
    // Each tile (64×64) = 16 blocks (4 rows per column-group)
    for (int block_row = 0; block_row < 4; block_row++) {   // 4 row groups per tile × 4 rows = 64 rows
        for (int block_col = 0; block_col < 1; block_col++) {  // 64 cols = 1 column group
            int block_idx = block_row * 1 + block_col;  // 0..15
            // Block: 4 rows starting at row_start + block_row*4, 64 cols
            uint64_t idx_base = (row_start + block_row * 4) * A->cols + col_start;
            // ... read the 144 bytes from the Q4_K tensor at this position ...
            memcpy(q4k_tile_out + block_idx * 144, A->data + offset, 144);
            // (offset calculation needs to account for Q4_K block boundaries in the original tensor)
        }
    }
}
```

### 7.4 Driver Register Changes

```verilog
// matmul_top.v - New control bit
localparam CTRL_MODE_Q4K = 1 << 6;  // In REG_CTRL_USER
```

### 7.5 C++ Caller Changes (tmac_gguf.cpp)

Replace `matmul_fpga_q8` call:

```cpp
// In forward_layer / matmul helper:
if (g_fpga_q4k && A->type == TENSOR_Q4_K) {
    matmul_fpga_q4k(A, x, y, rows, cols);
} else if (g_fpga_q8 && A->type == TENSOR_Q8_0) {
    matmul_fpga_q8(A, x, y, rows, cols);
} else {
    // fallback CPU matmul with dequant
}
```

---

## 8. File Changes Summary

### Verilog Files

| File | Change Type | Description |
|------|-------------|-------------|
| `matmul_q8_core.v` | **Major** | Add Q4_K mode: pre-dequant loading path, INT16×INT16 pipeline path, mode mux |
| `matmul_q8_top.v` | **Minor** | Add CTRL_MODE_Q4K bit, loading FSM extends to LP_Q4_DEQ phase |
| `matmul_q8_top.v` | Minor | Expand weight_buf range if needed for Q4_K data |

OR: **New file approach**

| File | Change Type | Description |
|------|-------------|-------------|
| `matmul_q4k_core.v` | **New** | Q4_K compute core (reuses portions of matmul_q8_core) |
| `matmul_q8_top.v` | **Minor** | Add instantiation of q4k_core, mode MUX |

The new file approach is cleaner — both cores coexist, top module selects.

### C++ Files

| File | Change Type | Description |
|------|-------------|-------------|
| `fpga_sim.hpp` | **Minor** | Add CTRL_MODE_Q4K constant |
| `fpga_sim.hpp` | Minor | Add `axi_vecmul_tile_q4k()` wrapper |
| `tmac_gguf.cpp` | **Major** | Add `matmul_fpga_q4k()` function |
| `tmac_gguf.cpp` | Minor | Add `--fpga-q4k` flag |

### Testbenches

| File | Change Type | Description |
|------|-------------|-------------|
| `tb_matmul_q8.v` | **Minor** | Add Q4_K test cases |
| `tb_cosim.v` | **Minor** | Add Q4_K tile dump format support |
| `tmac_gguf.cpp` | Minor | Add `--dump-tiles-q4k` flag |

---

## 9. Implementation Phases

### Phase 1: Software Simulation (verified Q4_K behavior)

1. Add `matmul_fpga_q4k()` to `tmac_gguf.cpp` (CPU simulation of FPGA behavior)
2. Add `axi_vecmul_tile_q4k()` to `fpga_sim.hpp`
3. Run with `--fpga-q4k` flag, compare output with Q4_K CPU reference
4. Verify throughput and accuracy

**Deliverable**: Working C++ simulation of Q4_K FPGA path

### Phase 2: Q4_K Core (new module)

1. Create `matmul_q4k_core.v`:
   - New wmem: 512 × 128-bit (2 BRAMs)
   - Loading FSM for Q4_K blocks → INT16 dequant
   - INT16×INT16 pipeline (same FSM pattern as Q8_0)
   - Mode signal (Q8_0 compatible interface sharing)
2. Test with tb_matmul_q8.v-like testbench (Q4_K test vectors)

**Deliverable**: Synthesizable Q4_K core module

### Phase 3: Top-level Integration

1. Instantiate both cores in `matmul_q8_top.v`
2. Add mode MUX to route signals
3. Add new AXI buffer for Q4_K weight data + scale data
4. Wire up new `REG_CTRL_USER` mode bits

**Deliverable**: Full top module with Q8_0 + Q4_K support

### Phase 4: Co-simulation & Verification

1. Extend `tb_cosim.v` for Q4_K tiles
2. Generate Q4_K tile dumps from `tmac_gguf --dump-tiles-q4k`
3. Run co-sim: 320 checks, all PASS
4. Compare accuracy vs CPU reference

**Deliverable**: Verified Q4_K FPGA path

### Phase 5: C++ Integration

1. Wire `--fpga-q4k` flag in `tmac_gguf.cpp`
2. Call `matmul_fpga_q4k()` for Q4_K tensors
3. Run full inference: 1 token, verify output matches CPU

**Deliverable**: End-to-end Qwen 2 0.5B inference via Q4_K FPGA

---

## 10. Estimated Timeline

| Phase | Effort | Dependencies |
|-------|--------|-------------|
| Phase 1 (Simulation) | 1 day | Understanding of Q4_K format |
| Phase 2 (Core) | 3 days | Phase 1 complete, fpga_sim reference |
| Phase 3 (Top) | 1 day | Phase 2 complete |
| Phase 4 (Co-sim) | 1 day | Phase 3 complete |
| Phase 5 (Integration) | 1 day | Phase 4 complete |

**Total**: ~7 days

---

## 11. Risk Assessment

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Float16 hardware dequant is timing-critical | High | Medium | Use UQ8.16 fixed-point approximation |
| 2 BRAMs for wmem exceed layout | Low | Low | Zynq 7010 has 60 BRAMs, using 2 of 60 |
| Q4_K loading phase adds startup latency | Medium | High | Acceptable: 8192 cycles vs 515 compute cycles |
| Q4_K→INT16 precision loss | Medium | Low | Use full-precision arithmetic, same as CPU |
| Pipeline timing closure at 150 MHz | Medium | Low | Same pipeline depth, wider data |

---

## 12. Appendix: Q4_K Block Indexing in FPGA Memory

### Q4_K Block → wmem Mapping

Within a 64×64 tile, the 16 Q4_K blocks (BI = 0..15) map to wmem addresses:

```
Block BI covers rows BR = (BI/1)*4 .. (BI/1)*4+3 (4 rows)
and columns 0..63 (all 64 columns in a tile)

Wait: 64 rows / 4 = 16 blocks. Yes: BI = row_group (0..15)

Within block BI:
  row_in_block = offset / 64  (0..3)
  col_in_block = offset % 64  (0..63)
  
  sub_block = offset / 32   (0..7)
    sub_block 0: row 0, col 0-31
    sub_block 1: row 0, col 32-63
    sub_block 2: row 1, col 0-31
    ...
    sub_block 7: row 3, col 32-63
  
  q4 value: from qs array at position sub_block
  sc, m: from scales array at position sub_block / 2
```

So for the pipeline iteration `{g, k}` where g = bank group (0..7), k = column (0..63):
```
Row group = g  (rows g*8 .. g*8+7)
  → This spans 2 Q4_K blocks: BI0 = g*2, BI1 = g*2+1
  → Block BI0 covers rows g*8..g*8+3
  → Block BI1 covers rows g*8+4..g*8+7

For row_in_bank (0..7, which is wi_i):
  block = g*2 + (row_in_bank >= 4 ? 1 : 0)
  row_in_block = row_in_bank % 4
  sub_block = row_in_block * 2 + (k >= 32 ? 1 : 0)
```

This mapping is needed for the loading FSM but NOT for the compute pipeline (since pre-dequantized values are stored linearly in wmem).
