#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>
#include <string.h>
#include "mann_kendall.h"

// for qsort
static int cmp(const void* a, const void* b) {
    float arg1 = *(const float*)a;
    float arg2 = *(const float*)b;
    if (arg1 < arg2) return -1;
    if (arg1 > arg2) return 1;
    return 0;
}

void mann_kendall(const float* data, size_t n_data, long long* S, float* var_S, float* Z){
    // This statistics test needs 4 elements
    if(n_data < 4){
        printf("Data point shall be greater than 4 points!");
        return;
    }
    
    // Compute the statictics value S
    long long s_val = 0;
    const int VEC_WIDTH = 8;
    for (size_t i=0; i<n_data-1; ++i){
        __m256 v_xi = _mm256_set1_ps(data[i]);
        size_t j = i + 1;

        // Memory address alignment fix
        size_t next_aligned_j = (j + VEC_WIDTH - 1) & ~(VEC_WIDTH - 1);
        size_t prologue_end = (next_aligned_j < n_data) ? next_aligned_j : n_data;
        for (; j < prologue_end; ++j) {
            if (data[j] > data[i]) s_val++;
            else if (data[j] < data[i]) s_val--;
        }

        // Start computation
        for (; j<=n_data-VEC_WIDTH; j+=VEC_WIDTH) {
            __m256 v_xj = _mm256_load_ps(&data[j]);
            __m256 gt_mask = _mm256_cmp_ps(v_xj, v_xi, _CMP_GT_OQ); // _CMP_GT_OQ -> a predicate, that is 'greater than'
            __m256 lt_mask = _mm256_cmp_ps(v_xj, v_xi, _CMP_LT_OQ);// _CMP_LT_OQ -> a predicate, that is 'less than'

            int gt_bits = _mm256_movemask_ps(gt_mask); // To combine the binary into a int
            int lt_bits = _mm256_movemask_ps(lt_mask); // To combine the binary into a int

            s_val += __builtin_popcount(gt_bits); // population count
            s_val -= __builtin_popcount(lt_bits); // population count
        }
        // Dealing with the rest elements
        for(; j < n_data; j++){
            if(data[j] > data[i])s_val++;
            else if(data[j] < data[i])s_val --;
        }
    }
    *S = s_val;

    // Compute variance of S
    float* sorted_data = (float*)malloc(sizeof(float)*n_data);
    memcpy(sorted_data, data, n_data*sizeof(float));
    qsort(sorted_data, n_data, sizeof(float), cmp);

    float tie_term = 0.0f;
    size_t i = 0;
    while (i < n_data) {
        size_t j = i + 1;
        while (j < n_data && sorted_data[j] == sorted_data[i])j++;
        size_t tie_count = j - i;
        if (tie_count > 1)tie_term += (float)tie_count * (tie_count - 1) * (2 * tie_count + 5);
        i = j;
    }
    free(sorted_data);

    float n_float = (float)n_data;
    float var_s_val = (n_float*(n_float-1)*(2*n_float + 5) - tie_term) / 18.0f;
    *var_S = var_s_val;

    // Compute the Z statistics
    if (var_s_val == 0) *Z = 0.0f;
    else {
        if (s_val > 0) *Z = ((float)s_val - 1.0) / sqrt(var_s_val);
        else if (s_val < 0) *Z = ((float)s_val + 1.0) / sqrt(var_s_val);
        else  *Z = 0.0;
    }
}