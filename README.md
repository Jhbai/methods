# Setup Enviroment
## 1. Simulator for STM32H7 ARMV6
### $ sudo apt install qemu-system-arm
It's Quick EMUlator, this is a open source of a simulator and virtualization software.
It simluates a hardware system, including
1. A ARM Cortext-M7
2. RAM
3. Peripherals, such as UART, Timer, Interrupt Controller
So, it can also simulate qemu-system-x86_64, qemu-system-mips ...etc.

### $ sudo apt install gcc-arm-none-eabi binutils-arm-none-eabi gdb-multiarch
1. gcc: GNU Compiler Collection
2. arm-none-eabi: this means the gcc target, none means "Bare Machine", eabi means "Embedded Application Binary Interface".
3. binutils: compile a program not only transform files into binary file, but a lot of tools used to dealing with binary files. including (a) "as: assembler", (b) "ld: linker", (c) "objcopy: sw that makes .elf to .bin", (d) "objdump: deassembly"
4. gdb-multiarch: gdb is a debugger, multiarch means multi-architecture.

## 2. A Project Structure
### startup.s
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
Notice that there is always a blank line in the very end of the file, since in linux/unix, a text file always strictly defined as a file that composed with several lines, there is a ending character \n in each lines.
A start-up code(Assembly), this is the first program MCU after powered, its tasks including
1. IVT Set up
2. Stack Pointer
3. Call the C main program

### app.c
    volatile unsigned int *const UART_DR = (unsigned int *)0x101f1000; // the address to match the 'versatilepb' board's UART0
    void print_str(const char *s){
	    while(*s != '\0'){
		    *UART_DR = *s;
		    s++;
	    }
    }

    int main(void){
	    print_str("This is the test code, print string via UART");
		while (1){};
	    return 0;
    }

    void UART0_Handler(void){
	    print_str("--- UART Interrupted Occured! --- \n");
    }
Notice that the main function shall not return in bare machine embedded development.
the UART0 address of UART0 in versatilepb is 0x101f1000

### stm32h7.ld
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
Notice that linker script use "space" to identify elements, so "ORIGIN = 0x08000000" each blanks is necessary.
In each segments(like .isr_vector, .text), a clever definition that each output segments shall be placed in which memory region is necessary.
the writing schema shall be <code>SECTION{SEGMENT{...} > MEMORY_LOCATION ... ...}</code>

## 3. Code Structure

## 4. Compile and Activate binary code
### $ arm-none-eabi-gcc -c -mcpu=arm926ej-s -o startup.o startup.s
### $ arm-none-eabi-gcc -c -mcpu=arm926ej-s -o app.o app.c
### $ arm-none-eabi-ld -T stm32h7.ld -o firmware.elf startup.o app.o
All this command can be turned into a Makefile:

	# The name of our final executable
	TARGET = firmware.elf
	
	# Compiler and tools
	CC = arm-none-eabi-gcc
	LD = arm-none-eabi-ld
	OBJCOPY = arm-none-eabi-objcopy
	
	# Compiler flags
	CFLAGS = -c -mcpu=cortex-m7 -mthumb -g
	LDFLAGS = -T stm32h7.ld
	
	# List of source files
	SOURCES_S = startup.s
	SOURCES_C = app.c
	
	# Generate list of object files from source files
	OBJECTS = $(SOURCES_S:.s=.o) $(SOURCES_C:.c=.o)
	
	# The default rule to build everything
	all: $(TARGET)
	
	# Rule to link the final executable
	$(TARGET): $(OBJECTS)
		$(LD) $(LDFLAGS) -o $@ $^
	
	# Rule to compile .s files into .o files
	%.o: %.s
		$(CC) $(CFLAGS) -o $@ $<
	
	# Rule to compile .c files into .o files
	%.o: %.c
		$(CC) $(CFLAGS) -o $@ $<
	
	# Rule to clean up generated files
	clean:
		rm -f $(OBJECTS) $(TARGET)
Then use <code>make</code> and <code>make clean</code> to compile code

### $ qemu-system-arm -M versatilepb -kernel firmware.elf -nographic
<code> ALSA </code> 是 <code> Advanced Linux Sound Architecture </code>, the driver of sound card in linux system
the architecture of versatilepb board is arm926ej-s
