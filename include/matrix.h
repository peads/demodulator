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
    #define HAS_EITHER_AVX_OR_SSE
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
    #define HAS_AARCH64
#endif

typedef void (*conversionFunction_t)(const void *__restrict__, const uint32_t, float *__restrict__);

#endif //DEMODULATOR_MATRIX_H
