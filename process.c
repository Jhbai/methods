#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <time.h>
#define WIDTH 1000
#define HEIGHT 1000
#define MAX_ITER 10000

int mandelbrot(double real, double imag) {
    double zr = 0.0, zi = 0.0;
    int iter = 0;
    while (zr * zr + zi * zi < 4.0 && iter < MAX_ITER) {
        double tmp = zr * zr - zi * zi + real;
        zi = 2.0 * zr * zi + imag;
        zr = tmp;
        iter++;
    }
    return iter;
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double start_time, end_time;
    start_time = MPI_Wtime();

    int rows_per_proc = HEIGHT / size;
    int start_row = rank * rows_per_proc;
    int end_row = (rank + 1) * rows_per_proc;

    for (int y = start_row; y < end_row; y++) {
        for (int x = 0; x < WIDTH; x++) {
            double real = (x - WIDTH / 2.0) * 4.0 / WIDTH;
            double imag = (y - HEIGHT / 2.0) * 4.0 / HEIGHT;
            mandelbrot(real, imag);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    if (rank == 0) {
        printf("[process] %f seconds\n", end_time - start_time);
    }

    MPI_Finalize();
    return 0;
}