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
