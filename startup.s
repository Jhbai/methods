@ startup.s for ARM926EJ-S on versatilepb

.global _start

.section .text
_start:
    @ Set up the stack pointer. 
    @ We use Supervisor mode, which is standard for bare-metal.
    ldr sp, =0x00100000  @ Stack grows downwards from 1MB mark

    @ Branch to our C code's main function
    bl  main

@ If main ever returns, loop forever here.
stop:
    b   stop
