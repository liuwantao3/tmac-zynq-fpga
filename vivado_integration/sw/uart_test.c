// uart_test.c - standalone USB-UART0 smoke test for MicroPhase Z7-Lite.
// Programs UART0 (0xE0000000, MIO 14/15, 115200 8N1) via direct xuartps
// register writes (see uart_init() in tmac_baremetal.h) and prints a
// banner + FPGA status. Validates the serial console without a model.
#include "tmac_baremetal.h"

int main(void)
{
    uart_init();
    uart_puts("\r\n=== tmac uart_test (Zynq-7010, UART0 115200 8N1) ===\r\n");

    uint32_t clk    = reg_read32(REG_CLK_CNT);
    uint32_t status = reg_read32(REG_STATUS);
    uint32_t dbg    = reg_read32(REG_DEBUG);

    uart_puts("FPGA CLK_CNT  = ");
    uart_puthex(clk);
    uart_puts("\r\nFPGA STATUS   = ");
    uart_puthex(status);
    uart_puts("\r\nFPGA DEBUG    = ");
    uart_puthex(dbg);
    uart_puts("\r\n");

    if (clk == 0) {
        uart_puts("ERROR: PL clock counter is 0 - FPGA not programmed?\r\n");
    } else {
        uart_puts("PL clock alive.\r\n");
    }

    uart_puts("Tick loop:\r\n");
    int i = 0;
    while (1) {
        uart_puts("tick ");
        uart_putdec(i);
        uart_puts("\r\n");
        i++;
        volatile unsigned int d;
        for (d = 0; d < 20000000; d++) {}
    }
    return 0;
}
