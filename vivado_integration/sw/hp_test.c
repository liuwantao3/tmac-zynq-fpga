#include "xil_printf.h"

#define DIAG_BASE 0x00203000

// PL310 L2 Cache Controller (Zynq-7010: 8-way, 512KB)
#define L2C_BASE       0xF8F02000
#define L2C_CLEAN_WAY  (L2C_BASE + 0x7B8)  // Write 0xFF to clean all 8 ways
#define L2C_CLEAN_INV  (L2C_BASE + 0x7FC)  // Write 0xFF to clean+inv all 8 ways
#define L2C_SYNC       (L2C_BASE + 0x730)  // Poll bit 0 until 0

static void l2_clean_all(void) {
    *(volatile unsigned*)L2C_CLEAN_WAY = 0xFF;
    __asm__("dsb");
    while (*(volatile unsigned*)L2C_SYNC & 1);
    __asm__("dsb");
    __asm__("isb");
}

static void l2_clean_inv_all(void) {
    *(volatile unsigned*)L2C_CLEAN_INV = 0xFF;
    __asm__("dsb");
    while (*(volatile unsigned*)L2C_SYNC & 1);
    __asm__("dsb");
    __asm__("isb");
}

// Flush ARM→DDR: L1 D-cache by MVA + L2 clean by way
static void flush_cache(unsigned long addr, unsigned long len) {
    unsigned long end = addr + len;
    for (unsigned long a = addr & ~31; a < end; a += 32)
        __asm__("mcr p15, 0, %0, c7, c10, 1" : : "r" (a));
    __asm__("dsb");
    l2_clean_all();
}

// Invalidate DDR→ARM: L2 clean+inv by way + L1 D-cache invalidate
static void inv_cache(unsigned long addr, unsigned long len) {
    (void)addr; (void)len; // full inv, ignore range
    l2_clean_inv_all();
    __asm__("mcr p15, 0, %0, c7, c6, 0" : : "r" (0)); // invalidate entire L1 D$
    __asm__("dsb");
    __asm__("isb");
}

#define IP_BASE 0x43C00000
#define REG_AP_CTRL  0x00
#define REG_STATUS   0x14
#define REG_WT_ADDR  0x18
#define REG_ACT_ADDR 0x1C
#define REG_RES_ADDR 0x20
#define REG_WR_TEST  0x28
#define REG_DEBUG    0x24

#define WT_ADDR   0x00200000
#define ACT_ADDR  (WT_ADDR + 8192)
#define RES_ADDR  (ACT_ADDR + 128)
#define WR_TEST_ADDR 0x00201000
#define OCM_MARK  0x00010000  // OCM marker for cache-free verification

static void wr(unsigned o, unsigned v) { *(volatile unsigned *)(IP_BASE+o) = v; }
static unsigned rd(unsigned o) { return *(volatile unsigned *)(IP_BASE+o); }

int main(void) {
    int i;
    xil_printf("\n=== HP1 test with L2 cache fix ===\n\r");

    // Write weight/act data (everyone=1, acts=1..64)
    for (int w = 0; w < 2048; w++)
        *(volatile unsigned *)(WT_ADDR + w*4) = 0x00010001;
    for (i = 0; i < 64; i++)
        *(volatile unsigned short *)(ACT_ADDR + i*2) = i+1;
    flush_cache(WT_ADDR, 8192);
    flush_cache(ACT_ADDR, 128);
    xil_printf("Weights/acts written and flushed\n\r");

    // AFI Experiment
    {
        unsigned *dbg = (unsigned*)(DIAG_BASE + 0x2D0);

        // 1. LOCKSTA before unlock
        dbg[0] = *(volatile unsigned*)0xF8000004;
        __asm__("dsb");

        // 2. Unlock SLCR
        *(volatile unsigned*)0xF8000008 = 0xDF0D;
        __asm__("dsb"); __asm__("isb");

        // 3. LOCKSTA after unlock
        dbg[1] = *(volatile unsigned*)0xF8000004;
        __asm__("dsb");

        // 4-5. AFI0_FIFO before/after write
        dbg[2] = *(volatile unsigned*)0xF8008004;
        *(volatile unsigned*)0xF8008004 = 0x00000707;
        __asm__("dsb");
        dbg[3] = *(volatile unsigned*)0xF8008004;

        // 6-9. AFI1 registers READ (before any ARM modification)
        dbg[4] = *(volatile unsigned*)0xF8009000;  // AFI1_CTRL
        dbg[5] = *(volatile unsigned*)0xF8009004;  // AFI1_FIFO_PARTITION
        dbg[6] = *(volatile unsigned*)0xF8009008;  // AFI1_WR_CHANNEL_CTRL
        dbg[7] = *(volatile unsigned*)0xF800900C;  // AFI1_RD_CHANNEL_CTRL

        // Enable AFI1 (in case ps7_post_config didn't)
        *(volatile unsigned*)0xF8009000 = 0x00000003;
        *(volatile unsigned*)0xF8009008 = 0x00000001;
        *(volatile unsigned*)0xF800900C = 0x00000001;
        __asm__("dsb");

        // 10-13. AFI1 registers AFTER enable
        dbg[8]  = *(volatile unsigned*)0xF8009000;
        dbg[9]  = *(volatile unsigned*)0xF8009004;
        dbg[10] = *(volatile unsigned*)0xF8009008;
        dbg[11] = *(volatile unsigned*)0xF800900C;

        // 14-15. AFI0 channel verify
        dbg[12] = *(volatile unsigned*)0xF8008008;
        dbg[13] = *(volatile unsigned*)0xF800800C;

        // 16-17. MIO test (verify SLCR unlock works)
        dbg[14] = *(volatile unsigned*)0xF8000240;
        *(volatile unsigned*)0xF8000240 = 0xDEADBEEF;
        __asm__("dsb");
        dbg[15] = *(volatile unsigned*)0xF8000240;

        // Lock SLCR
        *(volatile unsigned*)0xF8000008 = 0;
        __asm__("dsb"); __asm__("isb");

        flush_cache(DIAG_BASE + 0x2D0, 64);
        xil_printf("AFI done\n\r");
    }

    // === SPIN 1: DAP reads AFI experiment ===
    *(volatile unsigned*)(DIAG_BASE + 0x100) = 1;
    flush_cache(DIAG_BASE + 0x100, 4);
    xil_printf("SPIN1\n\r");
    while (*(volatile unsigned*)(DIAG_BASE + 0x100)) ;

    // === DIRECT WRITE TEST via HP1 ===
    // Init scratch with 0xDEADDEADDEADDEAD
    for (int wi = 0; wi < 8; wi++)
        *(volatile unsigned long long *)(WR_TEST_ADDR + wi*8) = 0xDEADDEADDEADDEADULL;
    flush_cache(WR_TEST_ADDR, 64);

    // Unlock SLCR again for AFI1 enable (ps7_post_config runs before ELF)
    *(volatile unsigned*)0xF8000008 = 0xDF0D;
    __asm__("dsb"); __asm__("isb");
    // Force-enable AFI1 write channel
    *(volatile unsigned*)0xF8009000 = 0x00000003;
    *(volatile unsigned*)0xF8009008 = 0x00000001;
    *(volatile unsigned*)0xF800900C = 0x00000001;
    *(volatile unsigned*)0xF8000008 = 0;
    __asm__("dsb"); __asm__("isb");

    // Start PL write test
    wr(REG_RES_ADDR, WR_TEST_ADDR);
    __asm__("dsb");
    wr(REG_WR_TEST, 1);
    __asm__("dsb");

    unsigned wt_done = 0;
    for (volatile int wd = 0; wd < 100000; wd++) {
        __asm__("nop");
        if (!(rd(REG_STATUS) & 1)) { wt_done = 1; break; }
    }
    wr(REG_WR_TEST, 0);

    // Invalidate + read back
    inv_cache(WR_TEST_ADDR, 64);
    volatile unsigned long long *wtest = (volatile unsigned long long *)WR_TEST_ADDR;
    for (int wi = 0; wi < 4; wi++) {
        unsigned lo = (unsigned)wtest[wi];
        unsigned hi = (unsigned)(wtest[wi] >> 32);
        *(volatile unsigned*)(DIAG_BASE + 0x240 + wi*8) = lo;
        *(volatile unsigned*)(DIAG_BASE + 0x244 + wi*8) = hi;
    }
    *(volatile unsigned*)(DIAG_BASE + 0x260) = wt_done;
    *(volatile unsigned*)(DIAG_BASE + 0x264) = rd(REG_DEBUG);
    flush_cache(DIAG_BASE + 0x240, 0x28);
    xil_printf("WR_TEST: done=%d\n\r", wt_done);

    // === SPIN 2: DAP reads write test results ===
    *(volatile unsigned*)(DIAG_BASE + 0x104) = 1;
    flush_cache(DIAG_BASE + 0x104, 4);
    xil_printf("SPIN2\n\r");
    while (*(volatile unsigned*)(DIAG_BASE + 0x104)) ;

    // === NORMAL COMPUTE TEST ===
    for (int ri = 0; ri < 64; ri++)
        *(volatile unsigned long long *)(RES_ADDR + ri*8) = 0xAAAAAAAAAAAAAAAAULL;
    flush_cache(RES_ADDR, 64*8);

    wr(REG_WT_ADDR, WT_ADDR);
    wr(REG_ACT_ADDR, ACT_ADDR);
    wr(REG_RES_ADDR, RES_ADDR);

    unsigned dbg0 = rd(REG_DEBUG);
    *(volatile unsigned*)(DIAG_BASE + 0x00) = dbg0;

    wr(REG_AP_CTRL, 1);

    int timeout = 5000000;
    unsigned dbg_at_start = rd(REG_DEBUG);
    *(volatile unsigned*)(DIAG_BASE + 0x04) = dbg_at_start;

    for (i = 0; i < timeout; i++) {
        unsigned st = rd(REG_STATUS);
        if (!(st & 1)) {
            unsigned dbg_done = rd(REG_DEBUG);
            unsigned st_done = rd(REG_STATUS);
            *(volatile unsigned*)(DIAG_BASE + 0x08) = 0xCAFE0004;
            *(volatile unsigned*)(DIAG_BASE + 0x0C) = dbg_done;
            *(volatile unsigned*)(DIAG_BASE + 0x10) = st_done;
            *(volatile unsigned*)(DIAG_BASE + 0x14) = i;
            break;
        }
        if (i == 0) {
            *(volatile unsigned*)(DIAG_BASE + 0x18) = rd(REG_DEBUG);
        }
    }
    if (i >= timeout) {
        unsigned dbg_to = rd(REG_DEBUG);
        *(volatile unsigned*)(DIAG_BASE + 0x1C) = 0xCAFE0005;
        *(volatile unsigned*)(DIAG_BASE + 0x20) = dbg_to;
    }

    // Read results with full cache invalidate
    inv_cache(RES_ADDR, 64*8);
    volatile unsigned long long *p = (volatile unsigned long long *)RES_ADDR;
    for (int ri = 0; ri < 16; ri++) {
        unsigned lo = (unsigned)p[ri];
        *(volatile unsigned*)(DIAG_BASE + 0x30 + ri*4) = lo;
    }

    // PS7 REGISTER DUMP
    *(volatile unsigned*)(DIAG_BASE + 0x270) = 0x50533700;
    __asm__("dsb");
    flush_cache(DIAG_BASE + 0x270, 4);
    __asm__("dsb");

    volatile unsigned *ps7_st = (volatile unsigned*)(DIAG_BASE + 0x274);
    *ps7_st = 0;

    volatile unsigned v;

    v = *(volatile unsigned*)0xF8000008; *ps7_st = 1;
    *(volatile unsigned*)(DIAG_BASE + 0x274) = v;

    v = *(volatile unsigned*)0xF8008000; *ps7_st = 2; *(volatile unsigned*)(DIAG_BASE + 0x278) = v;
    v = *(volatile unsigned*)0xF8008004; *ps7_st = 3; *(volatile unsigned*)(DIAG_BASE + 0x27C) = v;
    v = *(volatile unsigned*)0xF8008008; *ps7_st = 4; *(volatile unsigned*)(DIAG_BASE + 0x280) = v;
    v = *(volatile unsigned*)0xF800800C; *ps7_st = 5; *(volatile unsigned*)(DIAG_BASE + 0x284) = v;
    v = *(volatile unsigned*)0xF8008010; *ps7_st = 6; *(volatile unsigned*)(DIAG_BASE + 0x288) = v;
    v = *(volatile unsigned*)0xF8006094; *ps7_st = 7; *(volatile unsigned*)(DIAG_BASE + 0x28C) = v;

    v = *(volatile unsigned*)0xF8009000; *ps7_st = 10; *(volatile unsigned*)(DIAG_BASE + 0x2B4) = v;
    v = *(volatile unsigned*)0xF8009004; *ps7_st = 11; *(volatile unsigned*)(DIAG_BASE + 0x2B8) = v;
    v = *(volatile unsigned*)0xF8009008; *ps7_st = 12; *(volatile unsigned*)(DIAG_BASE + 0x2BC) = v;
    v = *(volatile unsigned*)0xF800900C; *ps7_st = 13; *(volatile unsigned*)(DIAG_BASE + 0x2C0) = v;
    v = *(volatile unsigned*)0xF8009010; *ps7_st = 14; *(volatile unsigned*)(DIAG_BASE + 0x2C4) = v;

    v = *(volatile unsigned*)0xF8006008; *ps7_st = 8; *(volatile unsigned*)(DIAG_BASE + 0x290) = v;
    *ps7_st = 9;

    __asm__("dsb");
    flush_cache(DIAG_BASE + 0x270, 0x60);
    __asm__("dsb");

    xil_printf("PS7 dumped\n\r");

    // DONE
    __asm__("dsb");
    *(volatile unsigned*)(DIAG_BASE + 0x200) = 0xCAFEBABE;
    flush_cache(DIAG_BASE + 0x200, 4);
    __asm__("dsb");
    *(volatile unsigned*)(DIAG_BASE + 0x204) = rd(REG_DEBUG);

    xil_printf("Done\n\r");
    while(1)__asm__("wfi");
}
