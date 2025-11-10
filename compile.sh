gcc -o thread thread.c -lpthread -lm
gcc -o serial serial.c -lm
mpicc -o process process.c -lm