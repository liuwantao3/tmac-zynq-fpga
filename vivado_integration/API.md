# FPGA Matmul Accelerator API

## Overview

The FPGA accelerator (Zynq 7010) offloads INT16×INT16 matmuls with Q8_0 and Q5_0
quantization via a descriptor-chain DMA engine:
- AXI4-Lite slave (GP0) for control/status at `0x43C0_0000`
- AXI HP0 read/write masters for bulk DDR transfers

A single descriptor can process multiple tiles (rows), reducing descriptor
chain length from thousands to ~12 per transformer layer. The FSM supports
Q8_0 (64×64 sub-tile, multi-column-group, multi-tile) and Q5_0 (4×896 tile,
multi-tile) compute paths, plus CPU_OP passthrough with optional interrupt-based
pause protocol.

---

## Register Map (AXI4-Lite `0x43C0_0000`)

| Offset | Name | Access | Description |
|--------|------|--------|-------------|
| `0x00` | `REG_START` | R/W | `[0]`: write 1 to start chain (auto-clears when FSM leaves IDLE) |
| `0x04` | `REG_CHAIN_CTRL` | R/W | `[0]`=resume, `[2]`=cpu_op_pending, `[3]`=intr_enable |
| `0x08` | `REG_GIE` | R/W | `[0]`: global interrupt enable |
| `0x0C` | `REG_ISR` | R/W | `[0]`: cpu_op_irq (W1C — write any value to clear) |
| `0x10` | `REG_Q8_NUM_GROUPS` | R/W | `[3:0]`: Q8 column groups fallback (0=single, 14=full 64×896) |
| `0x14` | `REG_STATUS` | R | `[8]`=rd_done, `[9]`=wr_done, `[15]`=busy |
| `0x18` | `REG_DESC_BASE` | R/W | Descriptor chain base DDR address (32-bit) |
| `0x1C` | `REG_DESC_TAIL` | R/W | Write 1 to enable chain (must be set) |
| `0x20` | `REG_DESC_HEAD` | R | Current descriptor index (auto-increments after each desc) |
| `0x28` | `REG_DEBUG` | R | Debug word (see bitfields below) |
| `0x2C` | `REG_CLK_CNT` | R | Free-running 32-bit clock cycle counter |
| `0x30` | `REG_CLK_CNT_SLOW` | R | Clock counter ÷ 1024 (for long timeouts) |
| `0x34` | `REG_ACT_INFO` | R | `act_addr` from last descriptor |
| `0x38` | `REG_DESC_INFO` | R | `{8'h0, act_total_bytes[23:0]}` from last descriptor |
| `0x3C` | `REG_Q8_DEBUG` | R | Q8 core debug (FSM state, core state, counters) |
| `0x40` | `REG_Q5_DEBUG` | R | Q5 core debug (see bitfields below) |
| `0x44` | `REG_Q5_DBG_CAP0` | R | `[31:16]`=core0_d_pre, `[15:0]`=core1_d_pre |
| `0x48` | `REG_Q5_DBG_CAP1` | R | `[31:16]`=core0_d_fp_lo, `[15:9]`=norm, `[8:6]`=core_state, `[5:0]`=blk_counter |
| `0x4C` | `REG_Q5_DBG_TRIG` | R/W | `[31:16]`=live d_pre, `[15:10]`=trig_blk, `[9]`=frozen, `[8]`=busy, `[0]`=arm |
| `0x50` | `REG_Q5_DBG_LIVE` | R | FSM state, tile, blk, core busy/frozen/state (live capture) |
| `0x54` | `REG_Q5_DBG_CAP2` | R | `core1.res1[31:0]` — low 32 bits of core 1 row 3 result |
| `0x58` | `REG_Q5_DBG_CAP3` | R | `core1.res1[47:32]`, `core1.res0[15:0]` |
| `0x5C` | `REG_Q5_DBG_CAP4` | R | `[31:27]`=core0_q5, `[15:0]`=core0_act_r |
| `0x60` | `REG_Q5_DBG_CAP5` | R | `[31:27]`=core1_q5, `[15:0]`=core1_act_r |
| `0x64` | `REG_Q5_DBG_SNAP` | R | Captured snapshot on arm/freeze (see bitfields) |
| `0x68` | `REG_Q5_DBG_WI_START` | R | `[31:27]`=c0_wi, `[26:22]`=c1_wi, `[5:0]`=blk (captured at first blk_entry) |

### REG_DEBUG (`0x28`) bitfields

| Bits | Field | Description |
|------|-------|-------------|
| `[31:27]` | `state` | FSM state (5-bit) — see state table below |
| `[26]` | `rd_done` | HP read done (sticky, clears on next LOAD_ACT entry) |
| `[25]` | `wr_done` | HP write done (sticky, clears on next WRITE_RES entry) |
| `[24]` | `rd_busy` | HP read master busy |
| `[23]` | `wr_busy` | HP write master busy |
| `[22]` | `q8_busy` | Q8 core busy |
| `[21:19]` | `wr_dbg_state` | Write master FSM state (3-bit) |
| `[18:16]` | `rd_dbg_state` | Read master FSM state (3-bit) |
| `[15]` | `q8_done` | Q8 core done (sticky) |
| `[14:11]` | `col_group` | Q8 column group counter (0–13) |
| `[10:8]` | `timeout_msb` | Timeout counter top 3 bits (`timeout_cnt[15:13]`) |
| `[7:0]` | `sc_byte_idx` | Scale byte counter |

### REG_Q8_DEBUG (`0x3C`) bitfields

| Bits | Field | Description |
|------|-------|-------------|
| `[31:27]` | `state` | FSM state (same as REG_DEBUG) |
| `[26]` | `q8_busy` | Q8 core busy |
| `[25]` | `q8_done` | Q8 core done (pulse) |
| `[24]` | `q8_start` | Q8 core start (pulse) |
| `[23]` | `q8_act_we` | Q8 act write enable (stuck-at-1 bug: never cleared after COPY_ACT_TO_CORE) |
| `[22:20]` | `q8_core_state` | Q8 core internal FSM state (3-bit: 0=IDLE, 1=CLEAR_ACC, 2=COMPUTE, 3=DRAIN, …, 7=DRAIN4) |
| `[19:17]` | `q8_core_g` | Q8 core bank counter (0..7) |
| `[16:11]` | `q8_core_k` | Q8 core column counter (0..63) |
| `[10:7]` | `misc` | `{copy_act_idx[1:0], q8_sc_we, sc_byte_idx[0]}` |
| `[6:0]` | `wt_byte_idx` | Weight byte index |

### REG_Q5_DEBUG (`0x40`) bitfields

| Bits | Field | Description |
|------|-------|-------------|
| `[31:27]` | `state` | FSM state |
| `[26]` | `rd_unpack_active` | Scale/block data unpack in progress |
| `[25]` | `q5_blk_valid` | Q5 core blk_valid pulse |
| `[24]` | `q5_any_busy` | Either core busy |
| `[23]` | `q5_busy_both` | Both cores busy |
| `[22]` | `q5_done0` | Core 0 done |
| `[21]` | `q5_done1` | Core 1 done |
| `[20:15]` | `q5_unpack_word` | Block data word counter (0..6) |
| `[14:9]` | `q5_blk_counter` | Block counter (0..55) |
| `[7:0]` | `rd_beat_cnt` | Read master AXI beats received |

### FSM States (REG_DEBUG[31:27] or REG_Q8_DEBUG[31:27])

| Value | Name | Description |
|-------|------|-------------|
| 0 | `IDLE` | Waiting for REG_START write |
| 1 | `FETCH_DESC` | Starting descriptor DDR read |
| 2 | `FETCH_DESC_W` | Waiting for descriptor read completion |
| 3 | `LOAD_ACT` | Starting activation data DDR read |
| 4 | `LOAD_ACT_W` | Waiting for activation read completion |
| 5 | `WRITE_RES` | Starting result DDR write |
| 6 | `WRITE_RES_W` | Waiting for result write completion |
| 7 | `DONE` | Chain complete (HEAD increments, checks next_addr) |
| 8 | `LOAD_WEIGHT` | Starting Q8 weight data DDR read (4096 bytes per group) |
| 9 | `LOAD_WEIGHT_W` | Waiting for weight read, writing 64-bit words to wmem |
| 10 | `LOAD_SCALES` | Starting Q8 scale data DDR read (256 bytes per group) |
| 11 | `LOAD_SCALES_W` | Waiting for scale read, unpacking pairs to smem |
| 12 | `COPY_ACT_TO_CORE` | Copying act_buf to core act_bram (64 × 16-bit) |
| 13 | `COMPUTE` | Pulsing q8_start to begin computation |
| 14 | `COMPUTE_W` | Waiting for Q8 core done |
| 15 | `READ_RES` | Reading Q8 results into act_buf (single-group) |
| 16 | `READ_RES_ACC` | Reading Q8 results, accumulating into acc_buf (multi-group) |
| 17 | `COPY_ACC_TO_BUF` | Copying acc_buf to act_buf for DDR writeback |
| 18 | `TIMEOUT_ERROR` | Timeout trap (latches `timeout_src` in REG_ACT_INFO[4:0], stalls) |
| 19 | `WRITE_RES_BURST` | Write result burst setup (computes burst addr/len) |
| 20 | `Q5_LOAD_NORM` | Starting Q5 row_norm DDR read (8 bytes = 4 × UQ8.8) |
| 21 | `Q5_LOAD_NORM_W` | Waiting for norm read + unpack to core norm BRAM |
| 22 | `Q5_COPY_ACT` | Starting Q5 activation DDR read (1792 bytes = 896 × INT16) |
| 23 | `Q5_COPY_ACT_W` | Waiting for act read + unpack to core act_mem BRAM |
| 24 | `Q5_BLOCK_COMPUTE` | Starting per-block DDR read (48 bytes = 12 AXI beats) |
| 25 | `Q5_BLOCK_COMPUTE_W` | Unpacking 6 rd_data words to core blk_d/qh/qs, pulsing blk_valid |
| 26 | `Q5_READ_RES` | Capturing res0/res1 from both cores via hierarchical refs |
| 27 | `CPU_OP_WAIT` | Interrupt mode: pulse desc_irq, set chain_ctrl[2], wait for resume |

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
- For Q8_0 with `num_tiles > 1`: activation is **reloaded** per tile.
  The FSM reads the same `act_addr` for each tile iteration.
- Q6_K/Q4_K dispatch (tensor_type 2,3 of hp_fsm_top) is not yet implemented
  in hp_fsm_top.v; those types are currently handled by the CPU fallback.
- CPU_OP (tensor_type=15) has two modes controlled by `chain_ctrl[3]`:
  - `chain_ctrl[3]=0` (default, backward compatible): simple passthrough
    (read act → write result). FSM flow: `FETCH_DESC → LOAD_ACT → WRITE_RES → DONE`.
  - `chain_ctrl[3]=1` (interrupt protocol): FSM enters `CPU_OP_WAIT` state,
    pulses `desc_irq`, sets `chain_ctrl[2]=1` (cpu_op_pending), and pauses.
    The CPU must: read `reg_desc_head` to identify the descriptor, perform
    the operation (RMSNorm, RoPE, SoftMax, etc.), write results to DDR,
    clear `reg_isr[0]`, and write `chain_ctrl[0]=1` to resume the chain.
    The FSM then clears resume and advances to the next descriptor.

---

## DDR Weight Layouts

### Q5_0 (tensor_type=1) — 4 rows × 896 cols per tile

Each tile is 2696 bytes. Tiles are contiguous in weight memory.

```
Tile i at weight_addr + i × 2696:

  offset  0:   block 0 (48 bytes)
  offset 48:   block 1 (48 bytes)
  ...
  offset 2640: block 55 (48 bytes)
  offset 2688: norm[0] UQ8.8   (2 bytes, little-endian)
  offset 2690: norm[1] UQ8.8
  offset 2692: norm[2] UQ8.8
  offset 2694: norm[3] UQ8.8
  offset 2696: start of next tile (if any)
```

Each block covers 32 Q5_0 quantized elements (32 columns). For each 4-row tile:
- **Blocks 0–27**: first row of each core (rows `core_id×2 + 0`)
- **Blocks 28–55**: second row of each core (rows `core_id×2 + 1`)

So blocks 0, 2, 4, …, 54 are row 0 and row 2 (one per core), while blocks
1, 3, 5, …, 55 are row 1 and row 3 — matching GGUF's row-major block layout
where each row of 896 columns = 28 Q5_0 blocks (896 ÷ 32).

The 896 activation values are read **once** from DDR and reused across all tiles.

### Per-block DDR layout (48 bytes = 12 × 32-bit words)

```
Word  0: [15:0]=core0_d_f16(0x3C00=1.0), [31:16]=core0_qh_low
Word  1: [15:0]=core0_qh_hi, [31:16]=core0_qs_word_low16
Words 2-4: core0_qs (3 × 32-bit = 96 bits of Q5 nibbles)
Word  5: [15:0]=core0_qs_word_low16, [31:16]=core1_d_f16(0x3C00)
Word  6: core1_qh (32-bit)
Words 7-10: core1_qs (4 × 32-bit = 128 bits of Q5 nibbles)
Word 11: padding (zero)
```

Total: 2 × (d[2] + qh[4] + qs[16]) + pad[4] = 48 bytes.<br>
The FSM reads 6 AXI beats (24 bytes each, 8-byte HP data width) per block:
6 × 8 = 48 bytes. Unpacked and pulsed to both cores via `blk_valid`.

### Q8_0 (tensor_type=0) — 64 rows × 896 cols (multi-group, multi-tile)

The Q8 core processes a 64×64 sub-tile per invocation. For 896 columns, the FSM
iterates 14 column groups, each loading new weights and scales and accumulating
into `acc_buf`. Per group:

```
  Group g at weight_addr + g × 4096:     4096 bytes INT8 column-major weights
                                        (64 rows × 64 bytes, 8 byte-lanes × 512 entries)
  Scale offset: weight_addr + N×4096 + g × 256
                                        (8 banks × 16 entries × 2 bytes = 256 bytes)
```

Where `N = num_groups` (from descriptor byte 20, or GP0 register fallback).

**Multi-group:** Set `num_groups=N` at descriptor byte `[20:3:0]` to iterate N
column groups (N×64 = total columns). For a full 896-column row: `num_groups=14`.
Each group loads its own weight slice and scale set from `weight_addr + g×4096`
and `weight_addr + N×4096 + g×256` respectively. Results accumulate across groups.

**Multi-tile:** Set `num_tiles` at descriptor bytes `[22:23]` to process multiple
64-row tiles in one descriptor. Each successive tile loads weight data from
`weight_addr + tile × tile_stride`, where `tile_stride = N×4096 + N×256`.
Results write to consecutive 512-byte blocks. **Activations are reloaded per tile**
(same `act_addr` for every tile — a known inefficiency).

### DDR address layout example (2 groups, 2 tiles)

```
Tile 0:
  weight_addr + 0       : group 0 weights (4096 bytes)
  weight_addr + 4096    : group 1 weights (4096 bytes)
  weight_addr + 8192    : group 0 scales (256 bytes)
  weight_addr + 8448    : group 1 scales (256 bytes)
  tile_stride = 4096×2 + 256×2 = 8704

Tile 1:
  weight_addr + 8704    : group 0 weights
  weight_addr + 12800   : group 1 weights
  weight_addr + 16896   : group 0 scales
  weight_addr + 17152   : group 1 scales
```

For `token_embd` (151936×896, Q8_0, 14 groups): `num_tiles=2374` processes the
full matrix in one descriptor.

### Q6_K (tensor_type=2) — 32 columns × 256 rows per tile

### Q4_K (tensor_type=3) — 56 columns × 256 rows per tile

---

## Tile Sizes

| Type | Tile (rows × cols) | Weight bytes/tile | Blocks | Act load | Desc per 896×896 | Desc per 4864×896 |
|------|--------------------|-------------------|--------|----------|-----------------|-----------------|
| Q5_0 (1 tile) | 4 × 896 | 2696 | 56 | 1792 B | 224 | 1216 |
| Q5_0 (multi-tile) | N×4 × 896 | N×2696 | N×56 | **once** 1792 B | **1** (N=224) | **1** (N=1216) |
| Q8_0 (1 group) | 64 × 64 | 4352 | — | 128 B | N/A (14 groups × 128B) | N/A |
| Q8_0 (multi-group) | 64 × 896 | N×4352 (N=14: 60928) | — | N×128 B | 1 desc (groups=14) | N/A |
| Q8_0 (multi-tile) | N×64 × 896 | N×tile_stride | — | **per tile** 1792 B | **1** (N=14, groups=14) | N/A |
| Q6_K | 32 × 256 | 6720 | 32 | — | N/A | 532 (CPU) |
| Q4_K | 56 × 256 | 8064 | 56 | — | N/A | 304 (CPU) |
| INT16 | 64 × 64 | 8192 | — | — | N/A | N/A |

Where `tile_stride = num_groups × 4352` (4096 weights + 256 scales per group).

All results are 48-bit S24.8 fixed-point, zero-extended to 64 bits (8 bytes/row).<br>
Act loads are included in total DDR read cost (affects throughput, not weight storage).

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

### Building descriptors

```c
// Generic descriptor write (32 bytes)
void write_desc(uint32_t *desc, uint32_t next, uint32_t weight, uint32_t act,
                uint32_t result, uint16_t tensor_type,
                uint8_t num_groups, uint16_t num_tiles, uint32_t act_bytes) {
    desc[0] = next;
    desc[1] = weight;
    desc[2] = act;
    desc[3] = result;
    desc[4] = tensor_type;                          // 0=Q8, 1=Q5, 15=CPU_OP
    desc[5] = (num_tiles << 16) | (num_groups & 0xFF);
    desc[6] = act_bytes;
    desc[7] = 0;
}

// Q5_0: attn_q (896×896, 224 tiles)
write_desc(desc, next_desc_addr, weight_ddr, act_ddr, result_ddr,
           1,       // tensor_type=Q5_0
           0,       // num_groups (unused by Q5)
           224,     // num_tiles = 896÷4
           1792);   // act_total_bytes = 896 × 2

// Q8_0: attn_v (128×896, 2 tiles, 14 groups)
write_desc(desc, next_desc_addr, weight_ddr, act_ddr, result_ddr,
           0,       // tensor_type=Q8_0
           14,      // num_groups = 14
           2,       // num_tiles = 128÷64
           1792);   // act_total_bytes

// CPU_OP: passthrough (read act_ddr → write result_ddr)
write_desc(desc, next_desc_addr, 0, act_ddr, result_ddr,
           15,      // tensor_type=CPU_OP
           0,       // num_groups (unused)
           1,       // num_tiles (unused)
           act_bytes);
```

### CPU_OP interrupt protocol (chain_ctrl[3]=1)

```c
// Enable interrupt mode before starting the chain:
reg_write(IP_BASE, REG_CHAIN_CTRL, 0x08);   // intr_enable (bit 3)
reg_write(IP_BASE, REG_GIE, 1);             // global interrupt enable

// Start chain normally
reg_write(IP_BASE, REG_DESC_BASE, ddr_desc_addr);
reg_write(IP_BASE, REG_DESC_TAIL, 1);
reg_write(IP_BASE, REG_START, 1);

// Interrupt handler (called when desc_irq fires):
void handle_fpga_irq(void) {
    uint32_t isr = reg_read(IP_BASE, REG_ISR);
    if (!(isr & 1)) return;

    uint32_t head = reg_read(IP_BASE, REG_DESC_HEAD);
    // Identify CPU_OP descriptor by head index
    // (CPU knows: "descriptor 0 → attn_norm, descriptor 4 → bias+rope+softmax")
    // Perform operation (RMSNorm, RoPE, SoftMax, etc.), write results to DDR
    // ...

    // Resume the FSM:
    reg_write(IP_BASE, REG_ISR, 1);                   // W1C: clear ISR[0]
    reg_write(IP_BASE, REG_CHAIN_CTRL, 0x09);          // set resume bit+keep intr_enable
}
```

### Restarting the chain

```c
reg_write(IP_BASE, REG_DESC_HEAD, 0);   // reset head (optional)
reg_write(IP_BASE, REG_START, 1);       // re-trigger (FSM goes IDLE → FETCH_DESC)
```

---

## Per-Layer Descriptor Chain Example (Qwen2-0.5B)

One layer uses **12 descriptors** (down from 6442 without multi-tile):

| # | Type | Act Source | Result Dest | Size | Notes |
|---|------|-----------|-------------|------|-------|
| 0 | CPU_OP | hidden | norm_out | 1792 B | RMSNorm (CPU) |
| 1 | Q5_0 | norm_out | q | 896×896 | num_tiles=224 |
| 2 | Q5_0 | norm_out | k | 896×896 | num_tiles=224 |
| 3 | Q8_0 | norm_out | v | 128×896 | num_tiles=2 (2×64 rows) |
| 4 | CPU_OP | — | context | — | bias+RoPE+softmax (CPU) |
| 5 | Q5_0 | context | attn_out | 896×896 | num_tiles=224 |
| 6 | CPU_OP | — | norm2 | — | residual+ffn_norm (CPU) |
| 7 | Q5_0 | norm2 | gate | 4864×896 | num_tiles=1216 |
| 8 | Q5_0 | norm2 | up | 4864×896 | num_tiles=1216 |
| 9 | CPU_OP | — | swiglu_out | — | SwiGLU (CPU) |
| 10 | CPU (Q6_K/Q4_K not in FSM) | swiglu_out | ffn_out | 896×4864 | ffn_down — CPU fallback |
| 11 | CPU_OP | — | hidden | — | residual (CPU) |

**FPGA compute breakdown per layer:**
- Q5_0: attn_q, attn_k, attn_output, ffn_gate, ffn_up — 5 matmuls
- Q8_0: attn_v — 1 matmul
- CPU: RMSNorm (×2), bias+RoPE+softmax, SwiGLU, residual (×2), ffn_down (Q6_K/Q4_K not yet in FSM) — 7 ops

CPU_OP descriptors (tensor_type=15) use `chain_ctrl[3]` to select mode:
- `chain_ctrl[3]=0`: simple passthrough (read act → write result, no interrupt)
- `chain_ctrl[3]=1`: interrupt protocol — FSM enters CPU_OP_WAIT, desc_irq fires,
  CPU does work and writes `chain_ctrl[0]=1` to resume. See CPU_OP_WAIT state above.

---

### Shared num_tiles / num_groups Fields

- `num_tiles` at descriptor bytes `[22:23]` (16-bit LE). Value `0` defaults
  to `1` for backward compatibility. Q5: 4-row tiles. Q8: 64-row tiles.
- `num_groups` at descriptor byte `[20:3:0]` (4-bit). Q8 only: number of
  64-column groups (14 for full 896 columns). Ignored by Q5_0. When `0`,
  falls back to `REG_Q8_NUM_GROUPS` (GP0 register `0x10`).
