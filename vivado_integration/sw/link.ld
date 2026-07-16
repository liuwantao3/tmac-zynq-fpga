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
        *(.text.*)
    } > DDR

    .rodata : {
        *(.rodata*)
        *(.rodata.*)
    } > DDR

    .data : {
        *(.data*)
    } > DDR

    .bss (NOLOAD) : {
        . = ALIGN(8);
        __bss_start = .;
        *(.bss*)
        *(COMMON)
        . = ALIGN(8);
        __bss_end = .;
    } > DDR

    .stack (NOLOAD) : {
        . = ALIGN(8);
        . = . + 0x10000;
        __stack_top = .;
    } > DDR

    /DISCARD/ : {
        *(.ARM.exidx*)
        *(.ARM.extab*)
        *(.comment*)
        *(.note*)
        *(.debug*)
    }
}
