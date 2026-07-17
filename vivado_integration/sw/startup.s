.syntax unified
.arch armv7-a
.arm

// =====================================================================
// Exception Vector Table (must be at _start, 8 words, 32 bytes total)
// =====================================================================
.section .text.boot
.global _start
.type _start, %function
_start:
    b reset_handler             // 0x00: Reset
    b undef_handler             // 0x04: Undefined Instruction
    b svc_handler               // 0x08: Supervisor Call (SWI)
    b pabt_handler              // 0x0C: Prefetch Abort
    b dabt_handler              // 0x10: Data Abort
    b .                         // 0x14: Reserved
    b irq_handler               // 0x18: IRQ
    b fiq_handler               // 0x1C: FIQ

// =====================================================================
// Reset Handler — CPU + FPU init, BSS clear, jump to main
// =====================================================================
reset_handler:
    // Enter SVC mode, disable IRQ/FIQ
    cpsid if
    mrs r0, cpsr
    bic r0, r0, #0x1F
    orr r0, r0, #0x13
    msr cpsr, r0

    // Set VBAR to _start (our vector table)
    ldr r0, =_start
    mcr p15, 0, r0, c12, c0, 0
    dsb
    isb

    // Disable MMU + caches
    mrc p15, 0, r0, c1, c0, 0
    bic r0, r0, #1
    bic r0, r0, #4
    mov r1, #0x1000
    bic r0, r0, r1
    mcr p15, 0, r0, c1, c0, 0
    dsb
    isb

    // Enable FPU (VFPv3-D16 on Cortex-A9)
    mrc p15, 0, r0, c1, c0, 2
    orr r0, r0, #(0xF << 20)
    mcr p15, 0, r0, c1, c0, 2
    dsb
    isb
    mov r3, #0x40000000
    vmsr fpexc, r3

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
    // Jump to main
    blx main

    // Should never return
    b .

// =====================================================================
// Exception Handlers
// =====================================================================
.section .text.exceptions,"ax",%progbits

// Data Abort — record fault info to 0x1F000800, then stall
.align 2
.type dabt_handler, %function
dabt_handler:
    ldr r0, =0x1F000810
    str lr, [r0]
    mrc p15, 0, r1, c6, c0, 0
    str r1, [r0, #4]
    mrc p15, 0, r2, c5, c0, 0
    str r2, [r0, #8]
    ldr r3, =0x44414254
    str r3, [r0, #-16]
1:  b 1b

// Prefetch Abort
.align 2
.type pabt_handler, %function
pabt_handler:
    ldr r0, =0x1F000810
    str lr, [r0, #12]
    ldr r3, =0x50414254
    str r3, [r0, #-16]
1:  b 1b

// Undefined Instruction
.align 2
.type undef_handler, %function
undef_handler:
    ldr r0, =0x1F000810
    str lr, [r0, #16]
    ldr r3, =0x554E4446
    str r3, [r0, #-16]
1:  b 1b

// SVC, IRQ, FIQ — unused, just return
.align 2
.type svc_handler, %function
svc_handler: bx lr
.type irq_handler, %function
irq_handler:   bx lr
.type fiq_handler, %function
fiq_handler:   bx lr
