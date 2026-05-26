#include "regs.h"

void uart_init(void) {
    // Disable UART
    reg_write(UART_BASE, UART_CR, 0x00000000);
    
    // Set mode: 8 bit, 1 stop, no parity, no flow control (mode reg = 0x00000100)
    // MR[1:0] = 10 for 8 bit, MR[6:4] = 010 for 1 stop, MR[7] = 0 for no parity
    reg_write(UART_BASE, UART_MR, 0x00000100);
    
    // Baud rate: assuming 50 MHz UART ref clock
    // BRGR = 50000000 / (8 * 115200) = 54.25, floor = 54
    reg_write(UART_BASE, UART_BRGR, 54);
    
    // Set RTS and DTR, enable TX and RX
    reg_write(UART_BASE, UART_CR, 0x00000100 | (1 << 5) | (1 << 4) | (1 << 2) | (1 << 1));
}

void uart_putc(char c) {
    // Wait for TX not full
    while (reg_read(UART_BASE, UART_SR) & UART_SR_TXFULL);
    reg_write(UART_BASE, UART_TX, c);
}

void uart_puts(const char *s) {
    while (*s) {
        if (*s == '\n')
            uart_putc('\r');
        uart_putc(*s++);
    }
}

void uart_puthex(uint32_t v, int digits) {
    for (int i = digits - 1; i >= 0; i--) {
        int nib = (v >> (i * 4)) & 0xF;
        uart_putc(nib < 10 ? '0' + nib : 'A' + nib - 10);
    }
}

void uart_putdec(int32_t v) {
    if (v < 0) {
        uart_putc('-');
        v = -v;
    }
    char buf[12];
    int i = 0;
    do {
        buf[i++] = '0' + v % 10;
        v /= 10;
    } while (v);
    while (i)
        uart_putc(buf[--i]);
}
