gcc -c -o theil_sen.o theil_sen.c -mavx # no linking, .o file generate only
gcc -c -o main.o main.c # no linking, .o file generate only
gcc -o main main.o theil_sen.o
./main