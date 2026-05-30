#include "regs.h"
#include "xil_printf.h"

#define NCOLS 64
#define NROWS 64

// Weight_buf packing:
// For entry = k*8+g (column k, group g=0..7), word offset c=0..3:
//   w = (k*8+g)*4 + c = k*32 + g*4 + c
//   weight_buf[w][15:0] = weight[k][g*8 + c*2]     (INT16)
//   weight_buf[w][31:16] = weight[k][g*8 + c*2+1]  (INT16)
void pack_weight_row(uint32_t *wb, int k, int row0, int row1) {
    // k = column, row0, row1 = two consecutive rows
    // Find g = row0 / 8, c = (row0 % 8) / 2
    int g = row0 / 8;
    int c = (row0 % 8) / 2;
    int w = k * 32 + g * 4 + c;
    int16_t v0 = (int16_t)(row0 * 64 + k);
    int16_t v1 = (int16_t)(row1 * 64 + k);
    wb[w] = ((uint32_t)(uint16_t)v1 << 16) | (uint16_t)v0;
}

// Build weight_buf from weight matrix W[col][row] = row*64 + col
static void build_weights(uint32_t *wb) {
    for (int k = 0; k < NCOLS; k++) {
        for (int g = 0; g < 8; g++) {
            for (int c = 0; c < 4; c++) {
                int row0 = g * 8 + c * 2;
                int row1 = row0 + 1;
                int w = k * 32 + g * 4 + c;
                int16_t v0 = (int16_t)(row0 * NCOLS + k);
                int16_t v1 = (int16_t)(row1 * NCOLS + k);
                wb[w] = ((uint32_t)(uint16_t)v1 << 16) | (uint16_t)v0;
            }
        }
    }
}

// Compute golden reference: result[row] = sum_{col} W[row][col] * act[col]
// Where W[col][row] = row*64 + col, act[col] = col + 1
static int32_t golden[NROWS];

static void compute_golden(void) {
    for (int r = 0; r < NROWS; r++) {
        int32_t sum = 0;
        for (int c = 0; c < NCOLS; c++) {
            sum += (int32_t)(r * NCOLS + c) * (int32_t)(c + 1);
        }
        golden[r] = sum;
    }
}

void main(void) {
    uint32_t wb[WT_SIZE];
    int pass = 1;

    // Build test weights and golden reference
    build_weights(wb);
    compute_golden();

    xil_printf("\n=== INT16 Matmul FPGA Test ===\n\r");

    // 1. Write weights (2048 words at 0x2000-0x3FFF)
    xil_printf("Writing weights... ");
    for (int i = 0; i < WT_SIZE; i++) {
        reg_write(IP_BASE, WT_BASE + i * 4, wb[i]);
    }
    xil_printf("OK\n\r");

    // 2. Write activations (64 INT16 at 0x1000-0x107C)
    xil_printf("Writing activations... ");
    for (int i = 0; i < 64; i += 2) {
        uint32_t pair = ((uint32_t)(uint16_t)(i + 2) << 16) | (uint16_t)(i + 1);
        reg_write(IP_BASE, ACT_BASE + i * 2, pair);
    }
    xil_printf("OK\n\r");

    // 3. Start computation
    xil_printf("Starting computation... ");
    reg_write(IP_BASE, REG_AP_CTRL, 1);
    xil_printf("started\n\r");

    // 4. Poll STATUS until done
    xil_printf("Waiting... ");
    uint32_t status;
    do {
        status = reg_read(IP_BASE, REG_STATUS);
    } while (status != STATUS_IDLE);
    xil_printf("done\n\r");

    // 5. Read results
    xil_printf("Reading results...\n\r");
    int errors = 0;
    for (int r = 0; r < NROWS; r++) {
        // Read lo 32 bits
        uint32_t lo = reg_read(IP_BASE, RES_LO_BASE + r * 4);
        // Read hi 16 bits
        uint32_t hi = reg_read(IP_BASE, RES_HI_BASE + r * 4);
        int64_t actual = ((int64_t)hi << 32) | (int64_t)lo;
        actual = actual & 0xFFFFFFFFFFFFLL;
        if (actual & 0x800000000000LL)
            actual |= ~0xFFFFFFFFFFFFLL;

        if (actual != (int64_t)golden[r]) {
            xil_printf("  Row %d: GOT %d EXPECT %d  FAIL\n\r", r, (int32_t)actual, golden[r]);
            errors++;
            pass = 0;
        }
    }

    if (pass) {
        xil_printf("\n=== ALL PASS ===\n\r");
    } else {
        xil_printf("\n=== %d FAILURES ===\n\r", errors);
    }

    // Halt
    while (1) {
        __asm__ volatile ("wfi");
    }
}
