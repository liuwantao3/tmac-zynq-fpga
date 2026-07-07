# Full Event Sequence: C++ → FPGA MatMul → C++

> **DEPRECATED (2026-07-07):** This documents the old `matmul_top.v` AXI-Lite
> buffer-based flow. The current active FSM is `hp_fsm_top.v` (HP descriptor-chain
> DMA). See `AGENTS.md` §"HP FSM flow" and `vivado_integration/API.md` for
> the current architecture and register map.

## Overview

This document describes the complete event sequence from when a C++ application calls the Q8_0 matrix multiplication accelerator, through data transfer and computation on the FPGA, back to result retrieval by C++.

---

## Phase 1: Configuration via AXI4-Lite (C++ → FPGA)

**1. C++ writes control registers:**
- `REG_AP_CTRL` = 0x0000_0004 (reset value, or configure)
- `REG_GIE` = 0x0000_0001 (enable global interrupts)
- `REG_IER` = 0x0000_0001 (enable interrupt 0)
- `REG_CTRL_USER` = 0x0000_0008 (set op_vecmul if needed)

Each write is an AXI4-Lite write transaction: `AWVALID/AWREADY` handshake, then `WVALID/WREADY` handshake, then `BVALID/BREADY` response.

---

## Phase 2: Data Transfer via AXI4-Lite (C++ → FPGA buffers)

**2. C++ writes 4096 bytes of weights** to address range `0x1000-0x1FFF`:
```
AXI addr: 0x1000, data: {w[3],w[2],w[1],w[0]}, strb: 0xF
AXI addr: 0x1004, data: {w[7],w[6],w[5],w[4]}, strb: 0xF
... 4096 total bytes
```
- `matmul_top.v` `write_buffer` task routes these to `weight_buf[0..4095]`

- **6. `matmul_top.v` detects `reg_ap_ctrl[0] && !core_busy && !loading`**:

C++                          FPGA (matmul_top)              FPGA (matmul_q8_core)
 │                                   │                                   │
 ├─── Write weights (4096 beats) ────→ weight_buf[]                       │
 ├─── Write scales (128 beats) ──────→ scale_buf[]                        │
 ├─── Write acts (64 beats) ─────────→ act_buf[]                          │
 ├─── Write AP_CTRL[0]=1 ────────────→ reg_ap_ctrl[0]                     │
 │                                   │                                   │
 │                                   ├── Auto-clear AP_CTRL[0]           │
 │                                   ├── loading=1, stream to core ──────→ wmem/smem/act_reg
 │                                   │         (4288 cycles)              │
 │                                   │                                   │
 │                                   ├── core_start=1 ──────────────────→ FSM: IDLE→COMPUTE
 │                                   │                                   │
 │                                   │                          ←── 515 cycles of pipeline ---→
 │                                   │                                   │
 │                                   │                          ←── core_done=1
 │                                   ├── draining=1, capture results     │
 │                                   │         (64 cycles)               │
 │                                   │                                   │
 ◅─── Read REG_STATUS ──────────────── reg_status=2                      │
 ◅─── Read result_buf[] ───────────── result_buf[0..63]                 │
 │                                                                   IDLE
```

---

## Address Map Summary

| Address Range | Size | Description |
|--------------|------|-------------|
| 0x0000-0x000F | 16B | Control registers (AP_CTRL, GIE, IER, ISR, CTRL_USER, STATUS) |
| 0x1000-0x1FFF | 4KB | Weight buffer (4096 bytes) |
| 0x2000-0x20FF | 256B | Scale buffer (128 × u16) |
| 0x2100-0x217F | 128B | Activation buffer (64 × i16) |
| 0x4000-0x40FF | 256B | Result buffer lower 32 bits (64 × u32) |
| 0x4200-0x427F | 256B | Result buffer upper 16 bits (64 × u16) |

---

## Performance Summary

| Phase | Cycles |
|-------|--------|
| Weight transfer | 4096 |
| Scale transfer | 128 |
| Activation transfer | 64 |
| Data loading to core | 4288 |
| Core computation (pipeline) | 515 |
| Result capture | 64 |
| **Total** | **~4867** |