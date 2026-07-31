/*
 * tmac_serial — bare-metal serial console test for MicroPhase Z7-Lite.
 *
 * Runs in the Vitis GUI (standalone BSP) over JTAG. Prints to the USB-UART0
 * (MIO 14/15, 115200 8N1) and dumps live FPGA registers via AXI4-Lite.
 *
 * Two UART paths are exercised:
 *   1. Direct xuartps register programming (same sequence as the fixed
 *      uart_init() in vivado_integration/sw/tmac_baremetal.h) — validates
 *      the serial fix on hardware.
 *   2. Standalone BSP driver via xil_printf().
 */
#include "xil_printf.h"
#include "xil_io.h"
#include "sleep.h"

#define UART0_BASE   0xE0000000UL
#define GP0_BASE     0x43C00000UL
#define REG_STATUS   0x14
#define REG_DEBUG    0x28
#define REG_CLK_CNT  0x2C
#define REG_ACT_INFO 0x34
#define REG_DESC_INFO 0x38
#define REG_Q8_DEBUG 0x3C

/* Direct xuartps register programming — identical to tmac_baremetal.h
 * uart_init(): CR=0x00, MR=0x04, BAUDGEN=0x18, BAUDDIV=0x34, SR=0x2C, FIFO=0x30. */
static void direct_uart_test(void)
{
    volatile uint32_t *uart = (volatile uint32_t *)UART0_BASE;

    uart[0] = 0x00000000;              /* CR: disable all */
    uart[0] = 0x00000001;              /* CR: RXRST */
    uart[0] = 0x00000002;              /* CR: TXRST */
    uart[0] = 0x00000000;              /* CR: idle */
    uart[1] = 0x00000020;              /* MR: 8N1, normal mode */
    uart[6] = 124;                     /* BAUDGEN (0x18) — 100 MHz ref clk */
    uart[13] = 6;                      /* BAUDDIV (0x34) */
    uart[0] = 0x00000014;              /* CR: RX_EN(0x04) | TX_EN(0x10) */

    const char *msg = "direct-register UART init OK (115200 8N1)\r\n";
    while (*msg) {
        char c = *msg++;
        if (c == '\n') {
            while (uart[11] & 0x10) {}
            uart[12] = '\r';
        }
        while (uart[11] & 0x10) {}     /* SR bit4 = TXFULL */
        uart[12] = (uint32_t)c;
    }
}

int main(void)
{
    direct_uart_test();

    xil_printf("\r\n=== tmac_serial (Zynq-7010, UART0 115200 8N1) ===\r\n");
    xil_printf("BSP xil_printf path OK\r\n");

    unsigned int clk    = Xil_In32(GP0_BASE + REG_CLK_CNT);
    unsigned int status = Xil_In32(GP0_BASE + REG_STATUS);
    unsigned int dbg    = Xil_In32(GP0_BASE + REG_DEBUG);
    unsigned int q8dbg  = Xil_In32(GP0_BASE + REG_Q8_DEBUG);
    unsigned int act    = Xil_In32(GP0_BASE + REG_ACT_INFO);
    unsigned int dsc    = Xil_In32(GP0_BASE + REG_DESC_INFO);

    xil_printf("FPGA: CLK_CNT=0x%08X STATUS=0x%04X DEBUG=0x%08X\r\n",
               clk, status & 0xFFFF, dbg);
    xil_printf("      Q8DBG=0x%08X ACT_INFO=0x%08X DESC_INFO=0x%08X\r\n",
               q8dbg, act, dsc);

    if (clk == 0)
        xil_printf("WARNING: CLK_CNT=0 - PL clock not running?\r\n");
    else
        xil_printf("PL clock alive.\r\n");

    unsigned int i = 0;
    while (1) {
        xil_printf("tick %u\r\n", i++);
        usleep(1000000);
    }
    return 0;
}
