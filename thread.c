#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include <time.h>
#define WIDTH 1000
#define HEIGHT 1000
#define MAX_ITER 10000
#define NUM_THREADS 4

pthread_mutex_t task_mutex;
int current_row = 0;

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

void *compute_mandelbrot() {
    int y;
    while (1) {
        pthread_mutex_lock(&task_mutex);
        if (current_row >= HEIGHT) {
            pthread_mutex_unlock(&task_mutex);
            break;
        }
        y = current_row;
        current_row++;
        pthread_mutex_unlock(&task_mutex);

        for (int x = 0; x < WIDTH; x++) {
            double real = (x - WIDTH / 2.0) * 4.0 / WIDTH;
            double imag = (y - HEIGHT / 2.0) * 4.0 / HEIGHT;
            mandelbrot(real, imag);
        }
    }
    pthread_exit(NULL);
}

int main() {
    pthread_t threads[NUM_THREADS];
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    pthread_mutex_init(&task_mutex, NULL);

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_create(&threads[i], NULL, compute_mandelbrot, NULL);
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    // Record the end time after all threads have finished
    clock_gettime(CLOCK_MONOTONIC, &end);

    pthread_mutex_destroy(&task_mutex);

    // Calculate the elapsed time
    double time_spent = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("[thread] %f seconds\n", time_spent);
    return 0;
}
