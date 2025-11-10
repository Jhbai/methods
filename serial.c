#include <stdio.h>
#include <stdlib.h>
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

int main() {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            double real = (x - WIDTH / 2.0) * 4.0 / WIDTH;
            double imag = (y - HEIGHT / 2.0) * 4.0 / HEIGHT;
            mandelbrot(real, imag);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            double real = (x - WIDTH / 2.0) * 4.0 / WIDTH;
            double imag = (y - HEIGHT / 2.0) * 4.0 / HEIGHT;
            mandelbrot(real, imag);
        }
    }

    // Calculate the elapsed time
    double time_spent = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("[serial] %f seconds\n", time_spent);
    return 0;
}
