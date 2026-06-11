.syntax unified
.arch armv7-a
.thumb

.section .text.boot
.global _start
.type _start, %function

_start:
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
    bl main

    // Should never return
    b .
