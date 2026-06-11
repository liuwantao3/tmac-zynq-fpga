ENTRY(_start)

MEMORY
{
    DDR  : ORIGIN = 0x00100000, LENGTH = 512M
}

SECTIONS
{
    .text : {
        KEEP(*(.text.boot))
        *(.text*)
    } > DDR

    .rodata : {
        *(.rodata*)
    } > DDR

    .data : {
        *(.data*)
    } > DDR

    .bss (NOLOAD) : {
        __bss_start = .;
        *(.bss*)
        *(COMMON)
        __bss_end = .;
    } > DDR

    .stack (NOLOAD) : {
        . = . + 0x10000;
        __stack_top = .;
    } > DDR

    /DISCARD/ : {
        *(.ARM.exidx*)
        *(.comment*)
        *(.note*)
    }
}
