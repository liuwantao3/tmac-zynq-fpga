#include "xil_printf.h"
#include "xil_mmu.h"

#define DIAG_BASE 0x00203000

// Clean+Invalidate cache to PoC (writes dirty data to DDR)
static void flush_cache(unsigned long addr, unsigned long len) {
    unsigned long end = addr + len;
    for (unsigned long a = addr & ~31; a < end; a += 32)
        __asm__("mcr p15, 0, %0, c7, c14, 1" : : "r" (a));
    __asm__("dsb");
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

static void wr(unsigned o, unsigned v) { *(volatile unsigned *)(IP_BASE+o) = v; }
static unsigned rd(unsigned o) { return *(volatile unsigned *)(IP_BASE+o); }

int main(void) {
    int i;
    xil_printf("\n=== HP write test (LVL_SHFTR_EN fix) ===\n\r");

    // Write weight/act data
    for (int w = 0; w < 2048; w++)
        *(volatile unsigned *)(WT_ADDR + w*4) = 0x00010001;
    for (i = 0; i < 64; i++)
        *(volatile unsigned short *)(ACT_ADDR + i*2) = i+1;
    flush_cache(WT_ADDR, 8192);
    flush_cache(ACT_ADDR, 128);
    xil_printf("Weights/acts written and flushed\n\r");

    // === FIX: Reconfigure SLCR page table (Strongly-Ordered) ===
    // Xilinx BSP maps 0xF8000000 with 0xC06 which is Device (C=0,B=1, shareable).
    // This is architecturally CORRECT for MMIO. Yet CPU reads return garbage.
    // Changing to 0xC02 (Strongly-Ordered, C=0,B=0) as a diagnostic step.
    // The real fix may be: re-writing entry + cache clean + TLB invalidate
    // to repair any runtime corruption of the page table.
    xil_printf("Fixing SLCR page table entry...\n\r");
    Xil_SetTlbAttributes(0xF8000000, 0xC02);
    __asm__("dsb");
    __asm__("isb");

    // Force D-cache clean of page table entry so L2 page table walker sees it
    unsigned ttbr0_val;
    __asm__("mrc p15,0,%[val],c2,c0,0" : [val] "=r" (ttbr0_val));
    unsigned pt_entry_addr = (ttbr0_val & 0xFFFFC000) + ((0xF8000000 >> 20) << 2);
    flush_cache(pt_entry_addr, 32);
    __asm__("dsb");

    // Also flush any stale cached lines for the SLCR address range
    flush_cache(0xF8000000, 0x100000);  // entire 1MB section
    __asm__("dsb");

    // TLB invalidate (redundant after Xil_SetTlbAttributes, but belt-and-suspenders)
    __asm__("mcr p15, 0, %0, c8, c7, 1" : : "r" (0xF8000000));
    __asm__("dsb");
    __asm__("isb");

    // Verify: read back page table entry
    unsigned pt_entry = *(volatile unsigned*)pt_entry_addr;
    xil_printf("TTBR0=0x%08X PT[F8000000]=0x%08X\n\r", ttbr0_val, pt_entry);
    *(volatile unsigned*)(DIAG_BASE + 0x2A0) = ttbr0_val;
    *(volatile unsigned*)(DIAG_BASE + 0x2A4) = pt_entry;
    __asm__("dsb");
    flush_cache(DIAG_BASE + 0x2A0, 8);
    __asm__("dsb");

    // === SLCR register writes ===
    // SLCR is mapped as Normal Write-Through by Xilinx BSP page table.
    // volatile + DSB must force stores to bus.
    // Readbacks are from same CPU path (not DAP) to verify writes took effect.

    unsigned slcr_lv, slcr_unlock, slcr_lv2;

    // Step 1: SLCR unlock
    slcr_unlock = *(volatile unsigned*)0xF8000008;
    *(volatile unsigned*)0xF8000008 = 0xDF0D;
    __asm__("dsb");
    __asm__("isb");
    slcr_lv = *(volatile unsigned*)0xF8000090;

    // Step 2: Enable LVL_SHFTR_EN (mask-bits[23:16] for data-bits[7:0])
    *(volatile unsigned*)0xF8000090 = (0x0F << 16) | 0x0F;  // 0x000F000F
    __asm__("dsb");
    __asm__("isb");
    slcr_lv2 = *(volatile unsigned*)0xF8000090;

    // Step 3: Deassert PL reset (mask-bits[31:16] for data-bits[15:0])
    unsigned fpga_rst_val = *(volatile unsigned*)0xF8000708;
    unsigned fpga_rst_data = fpga_rst_val & ~1;   // clear bit 0
    unsigned fpga_rst_mask = 1;                    // mask only bit 0
    *(volatile unsigned*)0xF8000708 = (fpga_rst_mask << 16) | fpga_rst_data;
    __asm__("dsb");
    __asm__("isb");
    unsigned fpga_rst_v2 = *(volatile unsigned*)0xF8000708;

    // Store SLCR probe values
    *(volatile unsigned*)(DIAG_BASE + 0x100) = slcr_unlock;    // before unlock
    *(volatile unsigned*)(DIAG_BASE + 0x104) = slcr_lv;        // before en
    *(volatile unsigned*)(DIAG_BASE + 0x108) = slcr_lv2;       // after en
    *(volatile unsigned*)(DIAG_BASE + 0x10C) = fpga_rst_val;   // before reset
    *(volatile unsigned*)(DIAG_BASE + 0x0F8) = 0xCAFEBABE;     // early marker
    *(volatile unsigned*)(DIAG_BASE + 0x204) = fpga_rst_v2;    // after reset
    __asm__("dsb");
    flush_cache(DIAG_BASE + 0x0F8, 8);    // flush early marker immediately
    flush_cache(DIAG_BASE + 0x100, 0x10); // flush SLCR probes (0x203100-0x20310F)
    flush_cache(DIAG_BASE + 0x204, 8);    // flush fpga_rst_v2
    __asm__("dsb");

    // Step 4: Reconfigure AFI0 FIFO partition (never properly tried with mask format!)
    // Reset: AFI0_FIFO_PARTITION=0x00000007 (RdFIFO=7, WrFIFO=0) — write FIFO dead.
    // No boot code writes to this in JTAG mode, so we are FIRST writer = write-once succeeds.
    // Use SLCR mask format: (mask << 16) | data, where mask[15:0] enables data[15:0].
    // Bits [11:8] = WrFIFO, bits[3:0] = RdFIFO.
    unsigned fifo_part_val = *(volatile unsigned*)0xF8008004;
    unsigned fifo_data = (6 << 8) | 2;   // WrFIFO=6, RdFIFO=2
    unsigned fifo_mask = (0xF << 8) | 0xF; // mask bits[11:8] and [3:0]
    *(volatile unsigned*)0xF8008004 = (fifo_mask << 16) | fifo_data;
    __asm__("dsb");
    __asm__("isb");
    unsigned fifo_part_v2 = *(volatile unsigned*)0xF8008004;

    // Markers to detect hang point
    *(volatile unsigned*)(DIAG_BASE + 0x120) = 0xBEEF0001;  // before AFI0_CTRL write
    __asm__("dsb");
    flush_cache(DIAG_BASE + 0x120, 4);
    __asm__("dsb");

    // Step 5: Enable AFI0 + bypass mode, then enable write channel
    *(volatile unsigned*)0xF8008000 = 0x00000003;  // AFI0_CTRL: enable + bypass
    __asm__("dsb"); __asm__("isb");
    *(volatile unsigned*)(DIAG_BASE + 0x124) = 0xBEEF0002;  // after AFI0_CTRL, before WRCHAN_CTRL
    __asm__("dsb");
    flush_cache(DIAG_BASE + 0x124, 4);
    __asm__("dsb");

    *(volatile unsigned*)0xF8008008 = 0x00000001;  // AFI0_WRCHAN_CTRL: enable write channel
    __asm__("dsb"); __asm__("isb");
    *(volatile unsigned*)(DIAG_BASE + 0x128) = 0xBEEF0003;  // after WRCHAN_CTRL
    __asm__("dsb");
    flush_cache(DIAG_BASE + 0x128, 4);
    __asm__("dsb");

    // Configure HP address map
    *(volatile unsigned*)0xF8008014 = 0x00000006;  // WRITE_ADDR: +200000 → 0x00200000
    *(volatile unsigned*)0xF8008018 = 0x00000010;  // WRITE_SIZE: 1MB
    *(volatile unsigned*)0xF800801C = 0x00000004;  // RD_ADDR: +200000 → 0x00200000
    *(volatile unsigned*)0xF8008028 = 0x00000000;  // RD_CTRL: 0
    __asm__("dsb");

    // Store FIFO partition probe values
    *(volatile unsigned*)(DIAG_BASE + 0x110) = fifo_part_val;  // before
    *(volatile unsigned*)(DIAG_BASE + 0x114) = fifo_part_v2;   // after
    *(volatile unsigned*)(DIAG_BASE + 0x204) = fpga_rst_v2;    // after reset
    __asm__("dsb");
    flush_cache(DIAG_BASE + 0x110, 8);
    flush_cache(DIAG_BASE + 0x204, 8);
    __asm__("dsb");

    __asm__("dsb");
    // SLCR lock (not strictly needed, but canonical)
    *(volatile unsigned*)0xF8000004 = 0x767B;
    __asm__("dsb");
    __asm__("isb");

    xil_printf("AXI_HP configured with write FIFO\n\r");

    // Init scratch with test pattern
    for (int wi = 0; wi < 8; wi++)
        *(volatile unsigned long long *)(WR_TEST_ADDR + wi*8) = 0xDEADDEADDEADDEADULL;
    flush_cache(WR_TEST_ADDR, 64);

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
    xil_printf("WR_TEST: done=%d\n\r", wt_done);
    *(volatile unsigned*)(DIAG_BASE + 0x260) = wt_done;
    *(volatile unsigned*)(DIAG_BASE + 0x264) = rd(REG_DEBUG);
    __asm__("dsb");
    flush_cache(DIAG_BASE + 0x260, 8);
    __asm__("dsb");

    // === NORMAL COMPUTE TEST (via HP) ===
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

    // Read results
    __asm__("dsb");
    flush_cache(RES_ADDR, 64*8);
    __asm__("dsb");
    volatile unsigned long long *p = (volatile unsigned long long *)RES_ADDR;
    for (int ri = 0; ri < 16; ri++) {
        unsigned lo = (unsigned)p[ri];
        *(volatile unsigned*)(DIAG_BASE + 0x30 + ri*4) = lo;
    }

    // Flush all DIAG data
    __asm__("dsb");
    flush_cache(DIAG_BASE + 0x00, 0x100);
    __asm__("dsb");

    // Write DONE marker + flush both words
    __asm__("dsb");
    *(volatile unsigned*)(DIAG_BASE + 0x200) = 0xCAFEBABE;
    *(volatile unsigned*)(DIAG_BASE + 0x204) = rd(REG_DEBUG);
    __asm__("dsb");
    flush_cache(DIAG_BASE + 0x200, 8);
    __asm__("dsb");

    xil_printf("Done\n\r");
    while(1)__asm__("wfi");
}
