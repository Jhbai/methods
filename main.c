#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mann_kendall.h"

float* align_malloc(size_t n){
	void* ptr = NULL;
	posix_memalign(&ptr, 32, sizeof(float)*n);
	return (float*)ptr;
}

void interpret_result(float z_stat, float alpha) {
    float z_critical = 1.96; // for alpha = 0.05
    if (alpha == 0.01) z_critical = 2.576;
    if (alpha == 0.1) z_critical = 1.645;
    
    printf("Under the alpha = %.2f (cirtical value ≈ %.3f):\n", alpha, z_critical);

    if (z_stat > z_critical) {
        printf("Reject Null Hypothesis, Data exists 'trending up'\n");
    } else if (z_stat < -z_critical) {
        printf("Reject Null Hypothesis, Data exists 'trending down'\n");
    } else {
        printf("Not reject Null Hypothesis, Data exists 'no trending'\n");
    }
}

int main() {
    const size_t DATA_SIZE = 1000;
    const float ALPHA = 0.05;

    // Using posix_memalign to align memory address
    float* data = align_malloc(DATA_SIZE);
    srand(time(NULL));
    long long s;
    float var_s, z;

    // Test 1
    {
        for (size_t i = 0; i < DATA_SIZE; ++i) 
            data[i] = 0.1 * i + (float)(rand() % 100) / 200.0; // y = 0.1*x + random_noise
        mann_kendall(data, DATA_SIZE, &s, &var_s, &z);
        printf("S = %.4lld\n", s);
        printf("Var(S) = %.4f\n", var_s);
        printf("Z = %.4f\n\n", z);
        interpret_result(z, ALPHA);
        printf("\n");
    }
    // Test 2
    {
        for (size_t i = 0; i < DATA_SIZE; ++i)
            data[i] = -0.1 * i + (float)(rand() % 100) / 200.0; // y = -0.1*x + random_noise
        mann_kendall(data, DATA_SIZE, &s, &var_s, &z);
        printf("S = %.4lld\n", s);
        printf("Var(S) = %.4f\n", var_s);
        printf("Z = %.4f\n\n", z);
        interpret_result(z, ALPHA);
        printf("\n");
    }

    // Test 3
    {
        for (size_t i = 0; i < DATA_SIZE; ++i) 
            data[i] = (float)(rand() % 1000) / 100.0;
        mann_kendall(data, DATA_SIZE, &s, &var_s, &z);
        printf("S = %.4lld\n", s);
        printf("Var(S) = %.4f\n", var_s);
        printf("Z = %.4f\n\n", z);
        interpret_result(z, ALPHA);
        printf("\n");
    }
    free(data);
    return 0;
}