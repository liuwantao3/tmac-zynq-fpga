#ifndef REGS_H
#define REGS_H

#include <stdint.h>

// AXI address map for axi_wrap_int16
// Base address set by Vivado block design (typically 0x43C0_0000)
#ifndef IP_BASE
#define IP_BASE 0x43C00000
#endif

// Control registers (0x0000-0x0FFF)
#define REG_AP_CTRL   0x00
#define REG_GIE       0x04
#define REG_IER       0x08
#define REG_ISR       0x0C
#define REG_CTRL_USER 0x10
#define REG_STATUS    0x14

// AP_CTRL bits
#define AP_START  0
#define AP_DONE   1
#define AP_IDLE   2
#define AP_READY  3

// STATUS values
#define STATUS_IDLE     0
#define STATUS_LOADING  1
#define STATUS_COMPUTE  2

// Act buffer write (0x1000-0x107C, up to 64 × INT16)
#define ACT_BASE 0x1000

// Weight buffer write (0x2000-0x3FFF, 2048 × 32-bit = 8192 bytes)
#define WT_BASE 0x2000
#define WT_SIZE 2048

// Result read (0x4000-0x40FC lo, 0x4200-0x427C hi)
#define RES_LO_BASE 0x4000
#define RES_HI_BASE 0x4200

// Act readback (0x5000-0x507C)
#define ACT_RD_BASE 0x5000

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
