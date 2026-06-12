#include "xil_printf.h"
#include "xparameters.h"

#define IP_BASE XPAR_AXI_WRAP_INT16_0_BASEADDR
#define WT_BASE  0x2000
#define WT_SIZE  2048
#define ACT_BASE 0x1000
#define RES_LO_BASE 0x4000
#define RES_HI_BASE 0x4200
#define REG_AP_CTRL 0x00
#define REG_STATUS  0x14
#define STATUS_IDLE 0

static void reg_write(unsigned int off, unsigned int val) {
    volatile unsigned int *p = (volatile unsigned int *)(IP_BASE + off);
    *p = val;
}

static unsigned int reg_read(unsigned int off) {
    volatile unsigned int *p = (volatile unsigned int *)(IP_BASE + off);
    return *p;
}

int main(void) {
    xil_printf("\n=== Diagnosis ===\n\r");

    // All weights = 1 (pack two INT16=1 per 32-bit word)
    xil_printf("Write all-1 weights... ");
    for (int i = 0; i < WT_SIZE; i++)
        reg_write(WT_BASE + i * 4, 0x00010001);
    xil_printf("OK\n\r");

    // Activations = 0..63
    xil_printf("Write activations... ");
    for (int i = 0; i < 64; i += 2) {
        unsigned pair = (unsigned)(i + 1) << 16 | (unsigned)i;
        reg_write(ACT_BASE + i * 2, pair);
    }
    xil_printf("OK\n\r");

    // Read back activations
    xil_printf("Act readback (0x5000):\n\r");
    unsigned v0 = reg_read(0x5000);
    unsigned v2 = reg_read(0x5004);
    xil_printf("  act[0]=%d act[1]=%d act[2]=%d act[3]=%d\n\r",
        v0 & 0xFFFF, v0 >> 16, v2 & 0xFFFF, v2 >> 16);

    // Compute
    xil_printf("Start... ");
    reg_write(REG_AP_CTRL, 1);

    unsigned int status;
    int timeout = 0;
    do {
        status = reg_read(REG_STATUS);
        if (++timeout > 1000000) { xil_printf("TIMEOUT!\n\r"); break; }
    } while (status != STATUS_IDLE);
    xil_printf("done\n\r");

    // Read results
    xil_printf("First 8 results:\n\r");
    for (int r = 0; r < 8; r++) {
        unsigned lo = reg_read(RES_LO_BASE + r * 4);
        unsigned hi = reg_read(RES_HI_BASE + r * 4);
        long long val = ((long long)hi << 32) | lo;
        val &= 0xFFFFFFFFFFFFLL;
        xil_printf("  Row %d: %lld\n\r", r, val);
    }

    xil_printf("Done.\n\r");
    while (1) __asm__ volatile ("wfi");
    return 0;
}
