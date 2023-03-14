//
// Created by Patrick Eads on 3/14/23.
//

#ifndef DEMODULATOR_DEMODULATOR_H
#define DEMODULATOR_DEMODULATOR_H

#include <stdio.h>
#include <stdint.h>
#include <immintrin.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <pthread.h>

//#define DEBUG

// sizeof(uint8_t)
#define INPUT_ELEMENT_BYTES 1
// sizeof(float)
#define OUTPUT_ELEMENT_BYTES 5
// sizeof(__m128)
#define MATRIX_ELEMENT_BYTES 16
#define VECTOR_WIDTH 4
#define LOG2_VECTOR_WIDTH 2
#define MAXIMUM_BUF_SIZE 1L << 33
#define DEFAULT_BUF_SIZE 16384

union m128_f {
    float buf[4];
    __m128 v;
};

struct rotationMatrix {
    const union m128_f a1;
    const union m128_f a2;
};

static const __m128 HUNDREDTH = {0.01f, 0.01f, 0.01f, 0.01f};
static const __m128 NEGATE_B_IM = {1.f,1.f,1.f,-1.f};
static const __m256i Z // all 127s
        = {0x7f7f7f7f7f7f7f7f, 0x7f7f7f7f7f7f7f7f, 0x7f7f7f7f7f7f7f7f, 0x7f7f7f7f7f7f7f7f};
static const __m128i FLOAT_ABS // all 0x7FFFFFFFUs
        = {9223372034707292159, 9223372034707292159};
//static const struct rotationMatrix PI_OVER_TWO_ROTATION = {
//        {0,-1,0,-1},
//        {1,0,1,0}
//};
//
//static const struct rotationMatrix THREE_PI_OVER_TWO_ROTATION = {
//        {0,1, 0,1},
//        {-1,0, -1,0}
//};
static const struct rotationMatrix CONJ_TRANSFORM = {
        {1, 0, 1, 0},
        {0, -1, 0, -1}
};

static uint8_t exitFlag = 0;
#endif //DEMODULATOR_DEMODULATOR_H
