#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "pelt.h"

void generate_test_data(float* data, size_t n) {
    for (size_t i = 0; i < n; i++) {
        float base_val = 0.0f;
        if (i >= 50 && i < 100) {
            base_val = 5.0f; // First Change Point
        }
        // 加上一點隨機雜訊
        float noise = ((float)rand() / (float)RAND_MAX - 0.5f) * 0.2f;
        data[i] = base_val + noise;
    }
}

int main(int argc, char** argv){
    // Init the parameters
    const size_t n_series = 150;
    const float penalty = 2.0f * logf((float)n_series);
    float* data = (float*)malloc(n_series * sizeof(float));

    // Malloc the result function
    int* changepoints = (int*)malloc(n_series * sizeof(int));
    int n_changepoints = 0;

    // Generate Sample Data
    generate_test_data(data, n_series);

    // Apply my algorithm
    pelt(data, n_series, penalty, changepoints, &n_changepoints);

    // Print Change Points
    printf("Found %d changepoints:\n", n_changepoints);
    for (int i = 0; i < n_changepoints; i++) {
        printf("  -> Index: %d\n", changepoints[i]);
    }
    /*
    -> Index: 50
    -> Index: 100
    -> Index: 150
    */
}