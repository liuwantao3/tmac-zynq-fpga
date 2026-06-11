#include "xil_printf.h"
#include "xparameters.h"
#include "sleep.h"

#define NCOLS 64
#define NROWS 64

// Weight buffer size (from regs.h)
#define WT_BASE  0x2000
#define WT_SIZE  2048
#define ACT_BASE 0x1000
#define RES_LO_BASE 0x4000
#define RES_HI_BASE 0x4200
#define REG_AP_CTRL 0x00
#define REG_STATUS  0x14
#define STATUS_IDLE 0
#define IP_BASE XPAR_AXI_WRAP_INT16_0_BASEADDR

static int32_t golden[NROWS];

static void reg_write(uint32_t off, uint32_t val) {
    volatile uint32_t *p = (volatile uint32_t *)(uintptr_t)(IP_BASE + off);
    *p = val;
}

static uint32_t reg_read(uint32_t off) {
    volatile uint32_t *p = (volatile uint32_t *)(uintptr_t)(IP_BASE + off);
    return *p;
}

static void compute_golden(void) {
    for (int r = 0; r < NROWS; r++) {
        int32_t sum = 0;
        for (int c = 0; c < NCOLS; c++)
            sum += (int32_t)(r * NCOLS + c) * (int32_t)(c + 1);
        golden[r] = sum;
    }
}

static void build_weights(uint32_t *wb) {
    for (int k = 0; k < NCOLS; k++)
        for (int g = 0; g < 8; g++)
            for (int c = 0; c < 4; c++) {
                int row0 = g * 8 + c * 2;
                int16_t v0 = (int16_t)(row0 * NCOLS + k);
                int16_t v1 = (int16_t)((row0 + 1) * NCOLS + k);
                wb[k * 32 + g * 4 + c] = ((uint32_t)(uint16_t)v1 << 16) | (uint16_t)v0;
            }
}

int main(void) {
    uint32_t wb[WT_SIZE];
    int pass = 1;

    build_weights(wb);
    compute_golden();

    xil_printf("\n=== INT16 Matmul FPGA Test ===\n\r");

    xil_printf("Writing weights... ");
    for (int i = 0; i < WT_SIZE; i++)
        reg_write(WT_BASE + i * 4, wb[i]);
    xil_printf("OK\n\r");

    xil_printf("Writing activations... ");
    for (int i = 0; i < 64; i += 2) {
        uint32_t pair = ((uint32_t)(uint16_t)(i + 2) << 16) | (uint16_t)(i + 1);
        reg_write(ACT_BASE + i * 2, pair);
    }
    xil_printf("OK\n\r");

    xil_printf("Starting computation... ");
    reg_write(REG_AP_CTRL, 1);
    xil_printf("started\n\r");

    xil_printf("Waiting... ");
    uint32_t status;
    int timeout = 0;
    do {
        status = reg_read(REG_STATUS);
        if (++timeout > 1000000) {
            xil_printf("TIMEOUT! STATUS=%d\n\r", status);
            break;
        }
    } while (status != STATUS_IDLE);
    if (status == STATUS_IDLE)
        xil_printf("done\n\r");

    xil_printf("Reading results...\n\r");
    int errors = 0;
    for (int r = 0; r < NROWS; r++) {
        uint32_t lo = reg_read(RES_LO_BASE + r * 4);
        uint32_t hi = reg_read(RES_HI_BASE + r * 4);
        int64_t actual = ((int64_t)hi << 32) | (int64_t)lo;
        actual = actual & 0xFFFFFFFFFFFFLL;
        if (actual & 0x800000000000LL)
            actual |= ~0xFFFFFFFFFFFFLL;

        if (actual != (int64_t)golden[r]) {
            xil_printf("  Row %d: GOT %d EXPECT %d  (lo=0x%08X hi=0x%04X) FAIL\n\r", r, (int32_t)actual, golden[r], lo, hi);
            errors++;
            pass = 0;
        }
    }

    if (pass)
        xil_printf("\n=== ALL PASS ===\n\r");
    else
        xil_printf("\n=== %d FAILURES ===\n\r", errors);

    while (1) {
        __asm__ volatile ("wfi");
    }
    return 0;
}
