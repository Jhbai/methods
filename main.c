#include <stdio.h>
#include <stdlib.h>
#include "theil_sen.h"

int main(int argc, char** argv){
  // Case 1
  size_t n = 9;
  float y[9] = {1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9};
  float x[9] = {3.2, 1.5, 0.8, 4.4, 3.3, 2.7, 6.6, 1.9, 2.4};
  float results[2];
  theil_sen(x, y, results, n);
  printf("\n(Model: y = %f + %f * x)\n", results[0], results[1]);

  // Case 2
  n = 10;
  float y2[10] = {1.1, 1.3, 1.9, 2.1, 3.0, 3.2, 3.9, 4.1, 5.0, 5.3};
  size_t win_size = 4;
  float* results2 = (float*)malloc(n * sizeof(float));
  theil_sen_window(y2, results2, n, win_size);
  printf("Original (y) -> Baseline (res)\n");
  for (size_t i = 0; i < n; i++) {
      printf("y[%02zu]: %5.2f -> res[%02zu]: %5.2f\n", i, y2[i], i, results2[i]);
  }


  return 0;
}