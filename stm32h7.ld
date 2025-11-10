/* linker.ld for versatilepb */
ENTRY(_start)

MEMORY
{
  RAM (rwx) : ORIGIN = 0x00000000, LENGTH = 128M
}

SECTIONS
{
    .text :
    {
        *(.text*)
    } > RAM

    .data :
    {
        *(.data*)
    } > RAM

    .bss :
    {
        *(.bss*)
    } > RAM
}
