.syntax unified
.arch armv7-a
.arm

.section .text.boot
.global _start
.type _start, %function

_start:
    // Enter SVC mode, disable IRQ/FIQ
    cpsid if
    mrs r0, cpsr
    bic r0, r0, #0x1F
    orr r0, r0, #0x13
    msr cpsr, r0

    // Disable MMU + caches before any C code runs
    mrc p15, 0, r0, c1, c0, 0  // Read SCTLR
    bic r0, r0, #1              // Clear M bit (MMU off)
    bic r0, r0, #4              // Clear C bit (D-cache off)
    mov r1, #0x1000
    bic r0, r0, r1              // Clear I bit (I-cache off)
    mcr p15, 0, r0, c1, c0, 0  // Write SCTLR
    dsb
    isb

    // Enable FPU (VFPv3-D16 on Cortex-A9)
    mrc p15, 0, r0, c1, c0, 2  // Read CPACR
    orr r0, r0, #(0xF << 20)   // Enable CP10 and CP11 (FPU access)
    mcr p15, 0, r0, c1, c0, 2  // Write CPACR
    dsb
    isb
    mov r3, #0x40000000
    vmsr fpexc, r3              // FPEXC.EN = 1 (enable FPU)

    // Set stack pointer
    ldr sp, =__stack_top

    // Clear BSS
    ldr r0, =__bss_start
    ldr r1, =__bss_end
    mov r2, #0
1:  cmp r0, r1
    bhs 2f
    str r2, [r0], #4
    b 1b
2:
    // Jump to main (BLX handles ARM/Thumb interworking)
    blx main

    // Should never return
    b .
