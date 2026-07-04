#ifndef REGS_H
#define REGS_H

#include <stdint.h>

// AXI address map for hp_fsm_top.v (base = 0x43C0_0000)
#ifndef IP_BASE
#define IP_BASE 0x43C00000
#endif

// HP FSM registers
#define REG_START         0x00  // [0]: write 1 to start chain (auto-clears)
#define REG_STATUS        0x14  // [8]=rd_done, [9]=wr_done, [15]=busy
#define REG_DESC_BASE     0x18  // descriptor chain base DDR address
#define REG_DESC_TAIL     0x1C  // tail index (write 1 to enable chain)
#define REG_DESC_HEAD     0x20  // current descriptor index (read-only)
#define REG_DEBUG         0x28  // debug status word (see bitfields below)
#define REG_CLK_CNT       0x2C  // free-running clock cycle counter
#define REG_CLK_CNT_SLOW  0x30  // clock counter divided by 1024
#define REG_ACT_INFO      0x34  // act_addr from last descriptor
#define REG_DESC_INFO     0x38  // {8'h0, act_total_bytes[23:0]}
#define REG_Q8_DEBUG      0x3C  // Q8 core debug word
#define REG_Q8_NUM_GROUPS 0x10  // [3:0]: column groups (0=single, 14=full 64x896)

// REG_DEBUG bitfields (matches hp_fsm_top.v:793-804)
#define DBG_STATE_MASK    0xF8000000  // [31:27] FSM state (5 bits)
#define DBG_RD_DONE       0x04000000  // [26] rd_done (sticky)
#define DBG_WR_DONE       0x02000000  // [25] wr_done (sticky)
#define DBG_RD_BUSY       0x01000000  // [24] rd_busy
#define DBG_WR_BUSY       0x00800000  // [23] wr_busy
#define DBG_Q8_BUSY       0x00400000  // [22] q8_busy
#define DBG_WR_DBG_STATE  0x00380000  // [21:19] write master FSM state
#define DBG_RD_DBG_STATE  0x00070000  // [18:16] read master FSM state
#define DBG_Q8_DONE       0x00008000  // [15] q8_done
#define DBG_COL_GROUP     0x00007800  // [14:11] col_group
#define DBG_TIMEOUT       0x00000700  // [10:8] timeout_cnt[15:13]
#define DBG_SC_BYTE_IDX   0x000000FF  // [7:0] sc_byte_idx

// REG_Q8_DEBUG bitfields
#define Q8DBG_STATE_MASK  0x1F000000  // [31:28] FSM state
#define Q8DBG_Q8_BUSY     0x08000000  // [27] q8_busy
#define Q8DBG_Q8_DONE     0x04000000  // [26] q8_done (pulse)
#define Q8DBG_Q8_START    0x02000000  // [25] q8_start (pulse)
#define Q8DBG_Q8_ACT_WE   0x01000000  // [24] q8_act_we (active during COPY_ACT_TO_CORE)
#define Q8DBG_Q8_RES_IDX  0x00FF0000  // [23:16] result read index (0..63)
#define Q8DBG_Q8_MISC     0x0000FF00  // [15:8] {copy_act_idx[2:0], q8_sc_we, sc_byte_idx[0], 3'b0}
#define Q8DBG_WT_BYTE_IDX 0x000000FF  // [7:0] weight byte index

// Memory-mapped I/O accessors
static inline void reg_write(uint32_t base, uint32_t off, uint32_t val) {
    volatile uint32_t *p = (volatile uint32_t *)(uintptr_t)(base + off);
    __asm__ volatile ("dsb" ::: "memory");
    *p = val;
    __asm__ volatile ("dsb" ::: "memory");
}

static inline uint32_t reg_read(uint32_t base, uint32_t off) {
    volatile uint32_t *p = (volatile uint32_t *)(uintptr_t)(base + off);
    __asm__ volatile ("dsb" ::: "memory");
    uint32_t v = *p;
    __asm__ volatile ("dsb" ::: "memory");
    return v;
}

#endif
