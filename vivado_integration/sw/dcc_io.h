#ifndef DCC_IO_H
#define DCC_IO_H

#include <stdint.h>

static inline void dcc_unlock(void) {
    uint32_t magic = 0xC5ACCE55;
    __asm__ volatile("mcr p14, 0, %0, c1, c0, 4" : : "r"(magic));
}

static inline void dcc_putc(char c) {
    uint32_t status;
    do {
        __asm__ volatile("mrc p14, 0, %0, c0, c1, 0" : "=r"(status));
    } while (status & (1 << 29));
    __asm__ volatile("mcr p14, 0, %0, c0, c5, 0" : : "r"((uint32_t)(unsigned char)c));
}

static inline void dcc_puts(const char *s) {
    while (*s) {
        if (*s == '\n') dcc_putc('\r');
        dcc_putc(*s++);
    }
}

static inline void dcc_puthex(uint32_t val) {
    static const char hex[] = "0123456789ABCDEF";
    dcc_putc('0'); dcc_putc('x');
    for (int i = 28; i >= 0; i -= 4) {
        dcc_putc(hex[(val >> i) & 0xF]);
    }
}

static inline void dcc_putdec(int val) {
    if (val < 0) { dcc_putc('-'); val = -val; }
    char buf[12]; int i = 0;
    if (val == 0) { dcc_putc('0'); return; }
    while (val) { buf[i++] = '0' + (val % 10); val /= 10; }
    while (i) dcc_putc(buf[--i]);
}

#endif
