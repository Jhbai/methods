#ifndef DTW_H
#define DTW_H

#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    float sakoe_chiba_band; // Pruning ratio
    float threshold; // Pruning threshold
} DTWConfig;

typedef struct {
    int start;
    int end;
    float distance;
    int path_length;
    int* path_i;
    int* path_j;
} DTWResult;
DTWResult* subsequence_alignment(const float* src, size_t src_len, const float* trg, size_t trg_len, const DTWConfig* config);
void dtw_free_result(DTWResult* result);
#ifdef __cplusplus
}
#endif
#endif // DTW_H