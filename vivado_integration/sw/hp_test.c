#include "xil_printf.h"

#define IP_BASE 0x43C00000
#define REG_AP_CTRL  0x00
#define REG_STATUS   0x14
#define REG_WT_ADDR  0x18
#define REG_ACT_ADDR 0x1C
#define REG_RES_ADDR 0x20

#define WT_ADDR   0x00200000
#define ACT_ADDR  (WT_ADDR + 8192)
#define RES_ADDR  (ACT_ADDR + 128)

static void wr(unsigned o, unsigned v) { *(volatile unsigned *)(IP_BASE+o) = v; }
static unsigned rd(unsigned o) { return *(volatile unsigned *)(IP_BASE+o); }

int main(void) {
    int i, r, c, k, g, wi, w;
    xil_printf("\n=== No cache ops ===\n\r");

    // Weights
    for (k = 0; k < 64; k++)
        for (g = 0; g < 8; g++)
            for (wi = 0; wi < 4; wi++) {
                w = k*32+g*4+wi;
                unsigned v = (unsigned short)((g*8+wi*2)*64+k)
                           | ((unsigned)(g*8+wi*2+1)*64+k) << 16;
                *(volatile unsigned *)(WT_ADDR + w*4) = v;
            }
    xil_printf("Weights: w[0]=0x%08X\n\r", *(volatile unsigned*)WT_ADDR);

    // Acts
    for (i = 0; i < 64; i++)
        *(volatile unsigned short *)(ACT_ADDR + i*2) = i+1;
    xil_printf("Acts: act[0]=%d\n\r", *(volatile unsigned short*)ACT_ADDR);

    // Configure + start
    wr(REG_WT_ADDR, WT_ADDR); wr(REG_ACT_ADDR, ACT_ADDR);
    wr(REG_RES_ADDR, RES_ADDR); wr(REG_AP_CTRL, 1);

    // Poll
    unsigned s = 1;
    for (i = 0; i < 20000000 && s; i++) s = rd(REG_STATUS);
    xil_printf("Done STATUS=%d polls=%d\n\r", s, i);
    xil_printf("DEBUG=0x%08X\n\r", rd(0x24));

    // Read result (no cache flush)
    volatile unsigned long long *p = (volatile unsigned long long *)RES_ADDR;
    long long v = (long long)(p[0] & 0xFFFFFFFFFFFFULL);
    if (v & 0x800000000000LL) v |= ~0xFFFFFFFFFFFFLL;
    xil_printf("Result[0]=%lld (hex 0x%08X_%08X)\n\r", v,
        (unsigned)(p[0]>>32), (unsigned)p[0]);

    // Check golden with a PRINT after the FPGA read
    int golden = 0;
    for (c = 0; c < 64; c++) golden += (0*64+c)*(c+1);
    xil_printf("golden[0]=%d (expect 87360)\n\r", golden);

    // Read debug
    unsigned dbg = rd(0x24);
    xil_printf("DEBUG=0x%08X (hp_rd=%d hp_wr=%d state=%d byte=%d)\n\r",
        dbg, (dbg>>31)&1, (dbg>>29)&1, (dbg>>20)&7, (dbg>>8)&0xFF);

    while(1)__asm__("wfi");
}
