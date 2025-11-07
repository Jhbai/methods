#ifndef THEIL_SEN_H
#define THEIL_SEN_H

#include <stdio.h>
#ifdef __cplusplus
extern "C"{
  #endif
  void theil_sen(float* x, float* y, float* res, size_t n);
  void theil_sen_window(float*y, float* res, size_t n, size_t win_size);
  #ifdef __cplusplus
}
#endif
#endif //THEIL_SEN_H
