# To remove all the elf
rm -rf build *.so machine_learning.c

# Start a gdb process and a thread to get into the code entry point for debugging
gdb --args python3 -B -m test