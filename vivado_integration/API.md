# FPGA Matmul Accelerator API

## Overview

The FPGA accelerator (Zynq 7010) offloads INT16×INT16 matmuls with 4 quantization
schemes. The `hp_fsm_top.v` module acts as a descriptor-chain DMA engine:
- AXI4-Lite slave (GP0) for control/status at `0x43C0_0000`
- AXI HP0 read/write masters for bulk DDR transfers

A single descriptor can process multiple 4-row tiles (Q5_0), reducing descriptor
chain length from thousands to ~5 per transformer layer.

---

## Register Map (AXI4-Lite `0x43C0_0000`)

| Offset | Name | Access | Description |
|--------|------|--------|-------------|
| `0x00` | `REG_START` | R/W | `[0]`: write 1 to start chain (auto-clears when FSM leaves IDLE) |
| `0x10` | `REG_Q8_NUM_GROUPS` | R/W | `[3:0]`: Q8 column groups (0=single, 14=full 64×896, fallback) |
| `0x14` | `REG_STATUS` | R | `[8]`=rd_done, `[9]`=wr_done, `[15]`=busy |
| `0x18` | `REG_DESC_BASE` | R/W | Descriptor chain base DDR address (32-bit) |
| `0x1C` | `REG_DESC_TAIL` | R/W | Write 1 to enable chain (unused—set to 1) |
| `0x20` | `REG_DESC_HEAD` | R | Current descriptor index (auto-increments after each desc) |
| `0x28` | `REG_DEBUG` | R | Debug word (see bitfields below) |
| `0x2C` | `REG_CLK_CNT` | R | Free-running 32-bit clock cycle counter |
| `0x30` | `REG_CLK_CNT_SLOW` | R | Clock counter ÷ 1024 (for long timeouts) |
| `0x34` | `REG_ACT_INFO` | R | `act_addr` from last descriptor |
| `0x38` | `REG_DESC_INFO` | R | `{8'h0, act_total_bytes[23:0]}` from last descriptor |
| `0x3C` | `REG_Q8_DEBUG` | R | Q8 core debug (FSM state, core state, counters) |

### REG_DEBUG (`0x28`) bitfields

| Bits | Field | Description |
|------|-------|-------------|
| `[31:27]` | `state` | FSM state (5-bit) — 0=IDLE, 7=DONE, 18=TIMEOUT_ERROR |
| `[26]` | `rd_done` | HP read done (sticky, clears on next LOAD_ACT entry) |
| `[25]` | `wr_done` | HP write done (sticky, clears on next WRITE_RES entry) |
| `[24]` | `rd_busy` | HP read master busy |
| `[23]` | `wr_busy` | HP write master busy |
| `[22]` | `q8_busy` | Q8 core busy |
| `[21:19]` | `wr_dbg_state` | Write master FSM state (3-bit) |
| `[18:16]` | `rd_dbg_state` | Read master FSM state (3-bit) |
| `[15]` | `q8_done` | Q8 core done (sticky) |
| `[14:11]` | `col_group` | Q8 column group counter (0–13) |
| `[10:8]` | `timeout_msb` | Timeout counter top 3 bits |
| `[7:0]` | `sc_byte_idx` | Scale byte counter |

### REG_Q8_DEBUG (`0x3C`) bitfields

| Bits | Field | Description |
|------|-------|-------------|
| `[31:27]` | `state` | FSM state (same as REG_DEBUG) |
| `[26]` | `q8_busy` | Q8 core busy |
| `[25]` | `q8_done` | Q8 core done (pulse) |
| `[24]` | `q8_start` | Q8 core start (pulse) |
| `[23]` | `q8_act_we` | Q8 act write enable |
| `[22:20]` | `q8_core_state` | Q8 core FSM state |
| `[19:17]` | `q8_core_g` | Q8 core bank counter |
| `[16:11]` | `q8_core_k` | Q8 core column counter |
| `[10:7]` | `misc` | `{copy_act_idx, q8_sc_we, sc_byte_idx[0]}` |
| `[6:0]` | `wt_byte_idx` | Weight byte index |

### FSM States (REG_DEBUG[31:27])

| Value | Name | Description |
|-------|------|-------------|
| 0 | `IDLE` | Waiting for REG_START write |
| 1 | `FETCH_DESC` | Starting descriptor DDR read |
| 2 | `FETCH_DESC_W` | Waiting for descriptor read completion |
| 3 | `LOAD_ACT` | Starting activation DDR read |
| 4 | `LOAD_ACT_W` | Waiting for activation read |
| 5 | `WRITE_RES` | Starting result DDR write |
| 6 | `WRITE_RES_W` | Waiting for result write |
| 7 | `DONE` | Chain complete |
| 8–17 | Q8 states | Weight/scale/act load, compute, result read |
| 18 | `TIMEOUT_ERROR` | Timeout trap (latches `timeout_src`, stalls) |
| 19 | `WRITE_RES_BURST` | Multi-burst result write |
| 20–26 | Q5 states | Norm/act load, block compute, result read |

---

## Descriptor Format (32 bytes)

Descriptors are stored in DDR as a linked list. Each descriptor is 32 bytes:

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| `0` | 4 | `next_addr` | DDR address of next descriptor (0 = end of chain) |
| `4` | 4 | `weight_addr` | DDR address of weight data (see DDR layout below) |
| `8` | 4 | `act_addr` | DDR address of activation data (INT16, row-major) |
| `12` | 4 | `result_addr` | DDR address for result writeback |
| `16` | 2 | `tensor_type` | 15=CPU_OP (passthrough), 0=Q8_0, 1=Q5_0. Q6_K(2) and Q4_K(3) dispatch planned. |
| `18` | 2 | _reserved_ | Zero |
| `20` | 1 | `num_groups` | Q8 column groups in bits [3:0] (0 falls back to REG_Q8_NUM_GROUPS) |
| `21` | 1 | _reserved_ | Zero |
| `22` | 2 | `num_tiles` | Number of tiles per descriptor (0→1 for backward compat). Q5_0: 4-row tiles. Q8_0: 64-row tiles. |
| `24` | 3 | `act_total_bytes` | Total activation bytes to read from DDR |
| `27` | 5 | _reserved_ | Zero |

**Notes:**
- All addresses are physical DDR byte addresses (32-bit, 4-byte aligned)
- `act_total_bytes` = `num_rows × num_cols × 2` (INT16 activations)
- For Q5_0 with `num_tiles > 1`: activation is read **once** from DDR.
  The FSM reuses the same activation for all tiles while streaming
  different weight rows.
- Q6_K and Q4_K dispatch (tensor_type 2,3) is planned but not yet implemented
  in the hp_fsm_top.v FSM; currently the `matmul_top.v` (PhaseB) path handles them.
- CPU_OP (tensor_type=15) in hp_fsm_top.v performs a simple passthrough
  (read act → write result). The interrupt-based pause protocol exists in
  `matmul_top.v` for future PhaseB integration.

---

## DDR Weight Layouts

### Q5_0 (tensor_type=1) — 4 rows × 896 cols per tile

Each tile is 2696 bytes. Tiles are contiguous in weight memory.

```
Tile i at weight_addr + i × 2696:

  offset  0:   block 0 (48 bytes: core0_d[2] + qh[4] + qs[16] + core1_d[2] + qh[4] + qs[16] + pad[4])
  offset 48:   block 1 (48 bytes)
  ...
  offset 2640: block 55 (48 bytes)
  offset 2688: norm[0] UQ8.8   (2 bytes, little-endian)
  offset 2690: norm[1] UQ8.8
  offset 2692: norm[2] UQ8.8
  offset 2694: norm[3] UQ8.8
  offset 2696: start of next tile (if any)
```

Each block covers 32 elements of the 896-column range. For each 4-row tile:
- Blocks 0–27: first row of each core (rows `core_id×2 + 0`)
- Blocks 28–55: second row of each core (rows `core_id×2 + 1`)

The 896 activation values are read once and shared across all tiles.

### Q8_0 (tensor_type=0) — 64 rows × 896 cols (multi-tile)

Weight data per tile (4352 bytes for 1 column group: 4096 weights + 256 scales):

```
  offset 0:    64 rows × 64 bytes = 4096 bytes (8 byte-lanes × 512 entries)
  offset 4096: scales: 8 banks × 16 entries × 2 bytes = 256 bytes
```

For `num_groups=N`: total per-tile stride = `4096 + N×256` bytes.
The Q8 core processes 64 columns at a time (one column group). A full 64×896
tile requires 14 column groups (896 ÷ 64 = 14). Each group iterates:
load scales → compute → accumulate. Set `num_groups=14` in descriptor byte 20.

**Multi-tile:** Set `num_tiles` in descriptor bytes [22:23] to process multiple
64-row tiles in one descriptor. Each successive tile loads weight data from
`weight_addr + tile × tile_stride`. Results write to consecutive 512-byte blocks.
Activations are reloaded per tile (same address as original `act_addr`).

For `token_embd` (151936×896, Q8_0): `num_tiles=2374` fits the whole matrix
in one descriptor, saving ~3.4 MB redundant DDR act reads per token.

### Q6_K (tensor_type=2) — 32 columns × 256 rows per tile

### Q4_K (tensor_type=3) — 56 columns × 256 rows per tile

---

## Tile Sizes

| Type | Tile (rows × cols) | Bytes/tile | Blocks | Desc per 896×896 | Desc per 4864×896 |
|------|--------------------|------------|--------|-----------------|-----------------|
| Q5_0 (1 tile) | 4 × 896 | 2696 + 1792 act | 56 | 224 | 1216 |
| Q5_0 (multi-tile) | N×4 × 896 | N×2696 + 1792 act | N×56 | **1** (N=224) | **1** (N=1216) |
| Q8_0 (1 tile) | 64 × 896 | 4352 | — | 2 (v: 128r) | N/A |
| Q8_0 (multi-tile) | N×64 × 896 | N×4352 + 1792 act | — | **1** (N=14) | N/A |
| Q6_K | 32 × 256 | 6720 | 32 | N/A | 532 |
| Q4_K | 56 × 256 | 8064 | 56 | N/A | 304 |
| INT16 | 64 × 64 | 8192 | — | N/A | N/A |

All results are 48-bit S24.8 fixed-point, zero-extended to 64 bits (8 bytes/row).

---

## C++ Usage

### Starting a descriptor chain

```c
#include "regs.h"

// 1. Write descriptor chain base address
reg_write(IP_BASE, REG_DESC_BASE, ddr_desc_addr);

// 2. Enable tail (must write 1)
reg_write(IP_BASE, REG_DESC_TAIL, 1);

// 3. Start the chain
reg_write(IP_BASE, REG_START, 1);
```

### Polling for completion

```c
// Poll descriptor HEAD
while (reg_read(IP_BASE, REG_DESC_HEAD) < expected_head) {
    // Optional: check STATUS for timeout or error
    uint32_t status = reg_read(IP_BASE, REG_STATUS);
    if (status & 0x10000) { /* TIMEOUT_ERROR — see DEBUG register */ }
}

// Or check STATUS busy bit
while (reg_read(IP_BASE, REG_STATUS) & 0x8000) { }
```

### Building a Q5_0 descriptor (attn_q: 896×896, 4 tiles)

```c
// Descriptor at ddr_desc_addr
uint32_t *desc = (uint32_t *)ddr_desc_addr;

desc[0] = next_desc_addr;           // chain pointer
desc[1] = weight_ddr_addr;          // weight_addr
desc[2] = act_ddr_addr;             // act_addr (896 × INT16)
desc[3] = result_ddr_addr;          // result_addr
desc[4] = 0x00000001;               // tensor_type=1, byte18=0, byte19=0
// byte20=0x00, byte21=0x00, byte22=0xE0(224), byte23=0x00 → copy as 32-bit:
desc[5] = (224 << 16) | (0 << 8) | 0;  // num_tiles=224
desc[6] = 1792;                     // act_total_bytes = 896 × 2
desc[7] = 0;                        // reserved

// Weight DDR layout: 224 tiles × 2696 bytes
for (int t = 0; t < 224; t++) {
    uint8_t *tile = weight_ddr_addr + t * 2696;
    // Write 56 blocks (48 bytes each)
    for (int b = 0; b < 56; b++) {
        write_q5_block(tile + b * 48, ...);
    }
    // Write 4 row_norm values at tile + 2688
    write_u16_le(tile + 2688, norm0);
    write_u16_le(tile + 2690, norm1);
    write_u16_le(tile + 2692, norm2);
    write_u16_le(tile + 2694, norm3);
}

// Activation: 896 × INT16 at act_ddr_addr
for (int i = 0; i < 896; i++)
    write_u16_le(act_ddr_addr + i * 2, activation[i]);

// Result buffer: 224 tiles × 4 rows × 8 bytes
memset(result_ddr_addr, 0, 224 * 32);
```

### Restarting the chain

```c
reg_write(IP_BASE, REG_DESC_HEAD, 0);   // reset head (optional)
reg_write(IP_BASE, REG_START, 1);       // re-trigger (FSM goes IDLE → FETCH_DESC)
```

---

## Per-Layer Descriptor Chain Example (Qwen2-0.5B)

One layer uses **5 descriptors** (down from 6442 without multi-tile):

| # | Type | Act Source | Result Dest | Size | Notes |
|---|------|-----------|-------------|------|-------|
| 0 | CPU_OP | hidden | norm_out | 1792 B | RMSNorm (CPU) |
| 1 | Q5_0 | norm_out | q | 896×896 | num_tiles=224 |
| 2 | Q5_0 | norm_out | k | 896×896 | num_tiles=224 |
| 3 | Q8_0 | norm_out | v | 128×896 | num_tiles=2 |
| 4 | CPU_OP | — | context | — | bias+RoPE+softmax (CPU) |
| 5 | Q5_0 | context | attn_out | 896×896 | num_tiles=224 |
| 6 | CPU_OP | — | norm2 | — | residual+ffn_norm (CPU) |
| 7 | Q5_0 | norm2 | gate | 4864×896 | num_tiles=1216 |
| 8 | Q5_0 | norm2 | up | 4864×896 | num_tiles=1216 |
| 9 | CPU_OP | — | swiglu_out | — | SwiGLU (CPU) |
| 10 | Q6_K/Q4_K | swiglu_out | ffn_out | 896×4864 | ffn_down |
| 11 | CPU_OP | — | hidden | — | residual (CPU) |

CPU_OP descriptors (tensor_type=15) in the current `hp_fsm_top.v` perform a
simple passthrough (read act → write result). The interrupt-based pause protocol
(where the CPU performs the operation and resumes the chain by clearing an ISR
and writing to CHAIN_CTRL) exists in `matmul_top.v` for future PhaseB integration.

---

### Shared num_tiles Field

For both Q5_0 and Q8_0, `num_tiles` is stored at descriptor bytes [22:23]
(16-bit little-endian). Value `0` defaults to `1` for backward compatibility.
The lower nibble of byte [20] stores `num_groups` (Q8 only, Q5_0 ignores it).
