#include <stdio.h>
#include <stdlib.h>
#include "theil_sen.h"

int main(int argc, char** argv){
  size_t n = 9;
  float y[9] = {1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9};
  float x[9] = {3.2, 1.5, 0.8, 4.4, 3.3, 2.7, 6.6, 1.9, 2.4};
  float results[2];
  theil_sen(x, y, results, n);
  printf("\n(Model: y = %f + %f * x)\n", results[0], results[1]);
  return 0;
}