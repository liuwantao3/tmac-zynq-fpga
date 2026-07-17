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

    // Set up exception vector table BEFORE anything that could fault.
    // Place vector table at _start (our code base, mapped to 0x00100000).
    // VBAR must point to our vector table (at offset 0 from _start).
    // The first 32 bytes of _start are: reset, undef, svc, pabt, dabt, rsvd, irq, fiq.
    // Load VBAR with this address.
    ldr r0, =_start
    mcr p15, 0, r0, c12, c0, 0  // Write VBAR
    dsb
    isb

    // Disable MMU + caches
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

    // Should never return — stale handler
    b .

// =====================================================================
// Exception Vectors (must be at _start, 8 words, 32 bytes total)
// Each entry is a 32-bit branch instruction to the handler.
// =====================================================================
.align 5  // Ensure aligned to 32 bytes (vector table boundary)
.global exception_vectors
exception_vectors:
    b _start                    // 0x00: Reset
    b undef_handler             // 0x04: Undefined Instruction
    b svc_handler               // 0x08: Supervisor Call (SWI)
    b pabt_handler              // 0x0C: Prefetch Abort
    b dabt_handler              // 0x10: Data Abort
    b .                         // 0x14: Reserved
    b irq_handler               // 0x18: IRQ
    b fiq_handler               // 0x1C: FIQ

// =====================================================================
// Exception Handlers — write fault info to DDR buffer at 0x1F000800,
// then stall.
// =====================================================================
.section .text.exceptions,"ax",%progbits

// Data Abort handler: records DFAR and DFSR, then stalls
.align 2
.type dabt_handler, %function
dabt_handler:
    // Save LR (abort link) to a known location
    ldr r0, =0x1F000810
    str lr, [r0]
    // Read Data Fault Address Register (DFAR)
    mrc p15, 0, r1, c6, c0, 0
    str r1, [r0, #4]
    // Read Data Fault Status Register (DFSR)
    mrc p15, 0, r2, c5, c0, 0
    str r2, [r0, #8]
    // Write magic "DABT" to mark data abort occurred
    ldr r3, =0x44414254
    str r3, [r0, #-16]          // at 0x1F000800
    // Stall
1:  b 1b

// Prefetch Abort handler
.align 2
.type pabt_handler, %function
pabt_handler:
    ldr r0, =0x1F000810
    str lr, [r0, #12]
    ldr r3, =0x50414254          // "PABT"
    str r3, [r0, #-16]
1:  b 1b

// Undefined Instruction handler
.align 2
.type undef_handler, %function
undef_handler:
    ldr r0, =0x1F000810
    str lr, [r0, #16]
    ldr r3, =0x554E4446          // "UNDF"
    str r3, [r0, #-16]
1:  b 1b

// SVC handler (unused)
.align 2
.type svc_handler, %function
svc_handler:
    bx lr

// IRQ handler (unused)
.align 2
.type irq_handler, %function
irq_handler:
    bx lr

// FIQ handler (unused)
.align 2
.type fiq_handler, %function
fiq_handler:
    bx lr
