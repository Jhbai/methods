gcc -c -o theil_sen.o theil_sen.c -mavx # no linking, .o file generate only
gcc -c -o main.o main.c # no linking, .o file generate only
gcc -o main main.o theil_sen.o
./main

# if want to compile shared object, you can use following commands: (PIC = Position Independent Code)
# gcc -fPIC -c theil_sen.c -mavx -o theil_sen.o
# gcc -shared -o libtheil_sen.so theil_sen.o
# gcc -o main main.c -L. -ltheil_sen (-L. means search from the current folder)
