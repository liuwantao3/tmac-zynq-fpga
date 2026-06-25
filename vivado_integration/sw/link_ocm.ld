OUTPUT_FORMAT("elf32-littlearm")
OUTPUT_ARCH(arm)
ENTRY(_start)

MEMORY
{
    OCM : ORIGIN = 0x00000000, LENGTH = 0x40000
}

SECTIONS
{
    . = 0x00000000;
    .text : { *(.text) *(.text.*) } > OCM
    .rodata : { *(.rodata) *(.rodata.*) } > OCM
    .data : { *(.data) *(.data.*) } > OCM
    .bss : { *(.bss) *(.bss.*) } > OCM
    /DISCARD/ : { *(.comment) *(.ARM.*) }
}
