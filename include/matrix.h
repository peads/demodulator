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

void noconversion(const void *__restrict__ in, uint32_t index, float *__restrict__ out);
void convertInt16ToFloat(const void *__restrict__ in, uint32_t index, float *__restrict__ out);
void convertUint8ToFloat(const void *__restrict__ in, uint32_t index, float *__restrict__ out);
float fastRsqrt(float y);
float slowRsqrt(float y);
#ifdef __AVX__
float intelRsqrt(float y);
#endif

#endif //DEMODULATOR_MATRIX_H
