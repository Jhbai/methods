#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include "dtw.h"

// ----- Auxilary function, using inline to enlarge memory consumption but decrease latency -----
static inline float min3(float a, float b, float c){
    float min = a;
    if(b < min) min = b;
    if(c < min) min = c;
    return min;
}

static inline int max(int a, int b){
    return (a > b)?a : b;
}

static inline int min(int a, int b){
    return (a > b)?b : a;
}

float euclidean_distance(float a, float b){
    return fabsf(a - b);
}

// Check sakoe-chiba band
static int is_in_band(int i, int j, int src_len, int trg_len, float band_ratio){
    if (band_ratio <= 0.0f) return 1; // no band using
    int band_width = (int)(band_ratio * src_len);
    if (band_width < 1) band_width = 1;

    // Compute the expectation position on the diagnosis
    float expected_j = (float)i * trg_len / src_len;
    return abs(j - (int)expected_j) <= band_width;
}

void dtw_free_result(DTWResult* result) {
    if (result) {
        if (result->path_i) free(result->path_i);
        if (result->path_j) free(result->path_j);
        free(result);
    }
}

// ----- Algorithm code -----
static DTWResult* dtw_distance(const float* src, const float* trg, size_t src_len, size_t start, size_t end, const DTWConfig* config){
    // Malloc the DTW distance matrix
    size_t win_len = end - start + 1; // begin and end points included
    float** cost = (float**)malloc((src_len + 1) * sizeof(float*)); // Notice that using n + 1 to let the index more intuitive
    for(size_t i = 0; i <= src_len; i++){
        cost[i] = (float*)malloc((win_len + 1)*sizeof(float));
    }

    // Initialization
    for(size_t i = 0; i <= src_len; i++)
        for(size_t j = 0; j <= win_len; j++)cost[i][j] = FLT_MAX;
    cost[0][0] = 0.0f;
    for(size_t i = 1; i <= win_len; i++)cost[0][i] = 0.0f; // columns, allow trg start from any position
    for(size_t i = 1; i <= src_len; i++)cost[i][0] = FLT_MAX; // Row, must match all the src elements

    float MinCost = FLT_MAX;
    int pruned = 0;

    // Main algorithm
    for(size_t i = 1; i <= src_len; i++){
        int valid = 0;
        for(size_t j = 1; j <= win_len; j++){
            // Check whether out of band
            if(!is_in_band(i-1, j-1, src_len, end - start + 1, config->sakoe_chiba_band)) continue; // [strcture entity] S.x; [structure] S->x

            // Compute the cost
            float dist = euclidean_distance(src[i-1], trg[start+j-1]);
            float prev_min = min3(cost[i-1][j], cost[i][j-1], cost[i-1][j-1]);
            if(prev_min != FLT_MAX){
                cost[i][j] = dist + prev_min;
                valid = 1;
                // If cost is too big(threshold defined beforehand), then discard this decision
                if(config->threshold > 0){
                    if(cost[i][j] > config->threshold){
                        cost[i][j] = FLT_MAX;
                        valid = 0;
                    }
                }
            }
        }
        if(!valid){
            pruned = 1;
            break;
        }
    }

    // Derive the best end point
    size_t best_end = 0;
    float best_cost = FLT_MAX;
    for(size_t i = 1; i <= win_len; i++){
        if(cost[src_len][i] < best_cost){
            best_cost = cost[src_len][i];
            best_end = i;
        }
    }

    // If no effective path, return NULL
    if(best_cost == FLT_MAX){
        for(size_t i = 0; i <= src_len; i++) free(cost[i]);
        free(cost);
        return NULL;
    }

    // Backtracing
    int* path_i = (int*)malloc(sizeof(int)*(src_len + win_len)); // The warping path for src
    int* path_j = (int*)malloc(sizeof(int)*(src_len + win_len)); // The warping path for trg
    int path_idx = 0;

    size_t i = src_len;
    size_t j = best_end;

    while(i>0){
        path_i[path_idx] = i -1;
        path_j[path_idx] = start + j - 1;
        path_idx++;
        
        float cost_diag = (j>0) ? cost[i-1][j-1] : FLT_MAX;
        float cost_up = cost[i-1][j];
        float cost_left = (j>0) ? cost[i][j-1] : FLT_MAX;

        if(cost_diag <= cost_up && cost_diag <= cost_left){
            i--; j--;
        }
        else if(cost_up <= cost_left){
            i--;
        }
        else j--;
    }

    // Construct the result
    size_t actual_start = path_j[path_idx - 1];
    DTWResult* result = (DTWResult*)malloc(sizeof(DTWResult));
    result->start = actual_start;
    result->end = start + best_end - 1;
    result->distance = best_cost;
    result->path_length = path_idx;

    result->path_i = (int*)malloc(sizeof(int) * path_idx);
    result->path_j = (int*)malloc(sizeof(int) * path_idx);

    // Reverse tracing
    for (int k = 0; k < path_idx; k++) {
        result->path_i[k] = path_i[path_idx - 1 - k];
        result->path_j[k] = path_j[path_idx - 1 - k];
    }

    // 清理
    free(path_i);
    free(path_j);
    for (size_t i = 0; i <= src_len; i++) free(cost[i]);
    free(cost);
    return result;
}

// Notice this is a heuristic implementation, not a global solution finding
DTWResult* subsequence_alignment(const float* src, size_t src_len, const float* trg, size_t trg_len, const DTWConfig* config){
    DTWResult* best_result = NULL;
    float best_distance = FLT_MAX;
    
    // Sliding window search
    size_t min_win = src_len;
    size_t max_win = min(src_len * 2, trg_len); // set the upper bound of matching length
    int found_good_match = 0;
    for(size_t i = min_win; i <= max_win && !found_good_match; i++){
        for(size_t start = 0; start <= trg_len - i; start++){
            size_t end = start + i - 1;
            DTWResult* result = dtw_distance(src, trg, src_len, start, end, config);
            if(result && result->distance < best_distance){
                if(best_result) dtw_free_result(best_result);
                best_result = result;
                best_distance = result->distance;
            }
            else if(result) 
                dtw_free_result(result);

            // Early stopping when find a good matching
            if(best_distance < config->threshold*0.5){
                found_good_match = 1;
                break;
            }
        }
    }
    return best_result;
}

