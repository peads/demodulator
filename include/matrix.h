//
// Created by peads on 4/3/23.
//

#ifndef DEMODULATOR_MATRIX_H
#define DEMODULATOR_MATRIX_H
#include <stdint.h>

#if (defined(__AVX__) || defined(__AVX2__))
    #define HAS_AVX
#endif

#if (defined(__SSE__) || defined(__SSE2__) || defined(__SSE3__) || defined(__SSE4_1__) \
        || defined(__SSE4_2__) || defined(__SSE_MATH__) || defined(__SSE2_MATH__) \
        || defined(__SSSE3__))
    #define HAS_SSE
#endif

#if defined(HAS_AVX) || defined(HAS_SSE)
    #define HAS_EITHER
#endif

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
