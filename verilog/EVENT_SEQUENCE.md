# Full Event Sequence: C++ → FPGA MatMul → C++

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
- `matmul_q8_top.v` `write_buffer` task (line 182) routes these to `weight_buf[0..4095]`

**3. C++ writes 256 bytes of scales** to address range `0x2000-0x20FF`:
```
AXI addr: 0x2000, data: {scale[1], scale[0]}, strb: 0xF
AXI addr: 0x2004, data: {scale[3], scale[2]}, strb: 0xF
... 128 scales total
```
- Routed to `scale_buf[0..127]` (16-bit entries)

**4. C++ writes 128 bytes of activations** to address range `0x2100-0x217F`:
```
AXI addr: 0x2100, data: {act[1], act[0]}, strb: 0xF
AXI addr: 0x2104, data: {act[3], act[2]}, strb: 0xF
... 64 activations total
```
- Routed to `act_buf[0..63]` (16-bit entries)

---

## Phase 3: Trigger Computation

**5. C++ writes to `REG_AP_CTRL` with bit[0]=1 (start):**
```cpp
// Pseudo-C++
*AP_CTRL = 0x0000_0001;  // Set start bit
```

---

## Phase 4: Auto-Clear & Data Loading (FPGA top module)

**6. `matmul_q8_top.v` detects `reg_ap_ctrl[0] && !core_busy && !loading`** (line 302):
- `reg_ap_ctrl[0]` is auto-cleared to 0 (line 304)
- `loading` flag set to 1 (line 305)
- `load_phase` starts at `LP_WEIGHT` (line 306)

**7. Loading state machine streams data to core** (lines 313-354):

| Phase | Core Signals | Beats |
|-------|-------------|-------|
| LP_WEIGHT | `wt_we=1`, `wt_addr`, `wt_din` | 4096 |
| LP_SCALE | `sc_we=1`, `sc_addr`, `sc_din` | 128 |
| LP_ACT | `act_we=1`, `act_addr`, `act_din` | 64 |

Each beat takes 1 clock cycle. Total loading: **4288 cycles**.

**8. Core receives data:**
- Weights written to BRAM `wmem[addr]` (line 41-54 in core)
- Scales written to distributed RAM `smem[addr]` (line 67-70 in core)
- Activations written to `act_reg[addr]` (line 77-80 in core)

---

## Phase 5: Core Computation Begins

**9. `core_start` is asserted** (line 291):
```verilog
assign core_start = reg_ap_ctrl[0] && !core_busy && !loading;
```

**10. Core FSM enters COMPUTE state** (matmul_q8_core.v lines 120-133):
- All 64 accumulators cleared to 0
- `busy` = 1
- `k=0`, `g=0`
- Transition to `COMPUTE`

---

## Phase 6: 3-Stage Pipeline Execution (512 iterations)

**Per iteration (one `{k, g}` pair):**

**Cycle N — Stage 0 (Address):**
- BRAM address `{g, k}` set on `wmem` port
- `act_reg[k]` and `smem[...+k[5]]` read
- Values captured into `p1_*` pipeline registers

**Cycle N+1 — Stage 1 (Dequant + Multiply):**
- `wmem_rdata` arrives (BRAM latency)
- 8 weights extracted: `wmem_rdata[wi_i*8 +: 8]`
- Each weight dequantized: `dequant_q8(weight, scale)` → INT16
- Multiplied by activation: `p1_act * dequantized`
- Results placed in `p2_partial[wi_i]`

**Cycle N+2 — Stage 2 (Accumulate):**
- `p2_partial` added to `acc[p2_row_base + wi_i]`
- `acc[base + i] <= acc[base + i] + partial` (read-before-write)

**Iteration ordering:**
```
k=0: g=0,1,2,3,4,5,6,7  (process 64 weights for col 0)
k=1: g=0,1,2,3,4,5,6,7  (process 64 weights for col 1)
...
k=63: g=0,1,2,3,4,5,6,7 (process 64 weights for col 63)
Total: 64 × 8 = 512 iterations
```

---

## Phase 7: Pipeline Drain

**After last iteration (k=63, g=7):**

**DRAIN state (1 cycle):**
- Stage 1 partials (if any) → Stage 2
- `p1_valid` cleared

**DRAIN2 state (1 cycle):**
- Stage 2 partials accumulated
- `core_done` = 1, `core_busy` = 0
- Return to IDLE

**Total cycles: 515 per tile** (1 exit + 512 compute + 1 drain + 1 drain2)

---

## Phase 8: Result Capture (core → top)

**11. `core_done` pulses** (line 199 in core)

**12. Top module's `draining` state machine activates** (lines 364-379):
- `draining` = 1
- `res_addr` cycles 0→63
- Each cycle: `result_buf[result_idx] <= res_dout` (line 373)
- After 64 cycles: `draining` = 0

---

## Phase 9: Interrupt (optional)

**13. If `reg_gie[0]` and `reg_ier[0]` set**, `reg_isr[0]` = 1 on `core_done` (line 410-411)

**14. `interrupt` output asserted** (line 418)

**15. C++ interrupt handler reads `REG_ISR` to confirm completion**

---

## Phase 10: C++ Reads Results

**16. C++ polls `REG_STATUS` or waits for interrupt**

**17. C++ reads result buffer** via AXI4-Lite reads:
```
AXI addr: 0x4000 → result_buf[0][31:0]
AXI addr: 0x4200 → result_buf[0][47:32] (upper 16 bits)
AXI addr: 0x4004 → result_buf[1][31:0]
... repeat for all 64 results
```

---

## Summary Timeline

```
C++                          FPGA (matmul_q8_top)              FPGA (matmul_q8_core)
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