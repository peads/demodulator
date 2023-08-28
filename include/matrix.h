//
// Created by peads on 4/3/23.
//

#ifndef DEMODULATOR_MATRIX_H
#define DEMODULATOR_MATRIX_H
#include <stdint.h>

typedef void (*conversionFunction_t)(const void *__restrict__, const uint32_t, float *__restrict__);
typedef float (*fastRsqrtFun_t)(float);

struct chars {
    uint8_t isRdc;      // 0
    uint8_t isOt;       // 1
    //uint8_t downsample; // 2
};

union fastRsqrtPun {
    uint32_t i;
    float f;
};

#endif //DEMODULATOR_MATRIX_H
