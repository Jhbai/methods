#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "dtw.h"

// Return config entity
DTWConfig create_config(float band, float threshold) {
    DTWConfig config;
    config.sakoe_chiba_band = band;
    config.threshold = threshold;
    return config;
}

void print_result(const char* test_name, DTWResult* result, const float* trg) {
    printf("--- %s ---\n", test_name);
    if (result == NULL) {
        printf("Result: NULL (No path found or pruned)\n");
    } else {
        printf("Result:\n");
        printf("  Distance:    %.4f\n", result->distance);
        printf("  Start index: %d\n", result->start);
        printf("  End index:   %d\n", result->end);
        printf("  Path Length: %d\n", result->path_length);
        
        printf("  Matched Subsequence (trg[%d...%d]):\n  ", result->start, result->end);
    }
    printf("----------------------------------------\n\n");
}

int main(int argc, char** argv) {
    DTWConfig config;
    DTWResult* result;
    // Test 1
    {
        float src[] = {1.0, 2.0, 3.0};
        float trg[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        config = create_config(0.0f, 0.0f); // 關閉 band 和 threshold
        result = subsequence_alignment(src, 3, trg, 6, &config);
        print_result("TEST 1: Perfect Match (Start)", result, trg); // 預期: dist=0.0, start=0, end=2
        dtw_free_result(result);
    }

    // Test 2
    {
        float src[] = {1.0, 2.0, 3.0};
        float trg[] = {4.0, 5.0, 6.0, 1.0, 2.0, 3.0};
        config = create_config(0.0f, 0.0f);
        result = subsequence_alignment(src, 3, trg, 6, &config);
        print_result("TEST 2: Perfect Match (End)", result, trg); // 預期: dist=0.0, start=3, end=5
        dtw_free_result(result);
    }

    // Test 3
    {
        float src[] = {1.0, 2.0, 3.0};
        float trg[] = {7.0, 8.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 9.0};
        config = create_config(0.0f, 0.0f);
        result = subsequence_alignment(src, 3, trg, 9, &config);
        print_result("TEST 3: Warped Match (Cost = 0)", result, trg); // 預期: dist=0.0, start=2, end=7
        dtw_free_result(result);
    }

    // Test 4
    {
        float src[] = {1.0, 2.0, 3.0};
        float trg[] = {100.0, 101.0, 102.0, 103.0};
        config = create_config(0.0f, 0.0f);
        result = subsequence_alignment(src, 3, trg, 4, &config);
        print_result("TEST 4: High Distance (No real match)", result, trg); // 預期: dist > 0 (會找到成本最低的爛匹配)
        dtw_free_result(result);
    }

    // Test 5
    {
        float src[] = {1.0, 2.0, 3.0};
        float trg[] = {5.0, 6.0, 1.1, 2.1, 3.1, 7.0}; // 成本 0.1+0.1+0.1 = 0.3
        config = create_config(0.0f, 0.2f); // Threshold 設為 0.2，比 0.3 低
        result = subsequence_alignment(src, 3, trg, 6, &config);
        print_result("TEST 5: Threshold Pruning (Should find NULL)", result, trg); // 預期: NULL
        dtw_free_result(result);
    }

    // Test 6
    {
        float src[] = {1.0, 2.0, 3.0};
        // 匹配 {1,2,3} 在 trg 中被嚴重拉伸，會超出窄帶
        float trg[] = {0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0}; 
        // 匹配路徑 (0,2), (1,5), (2,8) 偏離對角線
        config = create_config(0.1f, 0.0f); // 10% 窄帶 (band_width=1)
        result = subsequence_alignment(src, 3, trg, 9, &config);
        print_result("TEST 6: Sakoe-Chiba Pruning (Should find NULL or bad match)", result, trg); // 預期: dist > 0
        dtw_free_result(result);
    }

    // Test 10
    {
        float src[] = {1.0, 2.0, 3.0, 4.0, 5.0};
        float trg[] = {1.0, 2.0, 3.0};
        config = create_config(0.0f, 0.0f);
        result = subsequence_alignment(src, 5, trg, 3, &config);
        print_result("TEST 10: Target shorter than Source", result, trg); // 預期: NULL
        dtw_free_result(result);
    }

    // Test 11
    {
        float src[] = {1.0, 2.0, 3.0};
        float trg[] = {1.0, 2.0, 3.1,   9.9, 9.9,   1.0, 2.0, 3.0};
        // Threshold=0.3。提早停止條件: dist < 0.3 * 0.5 = 0.15
        config = create_config(0.0f, 0.3f); 
        result = subsequence_alignment(src, 3, trg, 8, &config);
        // 演算法會先找到 cost=0.1 的匹配。
        // 0.1 < 0.15，觸發提早停止。
        print_result("TEST 11: Early Stopping (Finds first 'good enough' match)", result, trg); // 預期: dist=0.1, start=0, end=2
        dtw_free_result(result);
    }

    // Test 12
    {
        float src[] = {1.0, 2.0, 3.0};
        float trg[] = {1.0, 2.0, 3.1,   9.9, 9.9,   1.0, 2.0, 3.0};
        // Threshold=0.0 關閉了剪枝和提早停止
        config = create_config(0.0f, 0.0f); 
        result = subsequence_alignment(src, 3, trg, 8, &config);
        // 演算法會完整搜尋，並找到 cost=0.0 的最佳匹配
        print_result("TEST 12: Disabled Early Stopping (Finds global best match)", result, trg); // 預期: dist=0.0, start=5, end=7
        dtw_free_result(result);
    }

    printf("All tests completed.\n");
    return 0;
}