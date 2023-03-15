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
// sizeof(float) >> 1
#define OUTPUT_ELEMENT_BYTES 4
// sizeof(__m128)
#define MATRIX_ELEMENT_BYTES 16
#define MATRIX_WIDTH 4
#define LOG2_VECTOR_WIDTH 2
#define MAXIMUM_BUF_SIZE 8589934592
#define DEFAULT_BUF_SIZE 1024
//(MATRIX_WIDTH << 20)
//1048576

union m128_f {
    float buf[4];
    __m128 v;
};

struct rotationMatrix {
    const union m128_f a1;
    const union m128_f a2;
};

struct readArgs {
    char *inFile;
    char *outFileName;
    uint8_t downsample;
    uint8_t isRdc;
    uint8_t isOt;
    __m128 *squelch;
    __m128 *buf;
    uint64_t len;
    FILE *outFile;
};

//static const __m128 HUNDREDTH = {0.01f, 0.01f, 0.01f, 0.01f};
static const __m128 NEGATE_B_IM = {1.f,1.f,1.f,-1.f};
static const __m64 Z = {0x7f7f7f7f7f7f7f7f}; // all 127s
        //= {-127.f, -127.f, -127.f, -127.f};//{0x7f7f7f7f7f7f7f7f, 0x7f7f7f7f7f7f7f7f, 0x7f7f7f7f7f7f7f7f, 0x7f7f7f7f7f7f7f7f};
//static const __m128i FLOAT_ABS // all 0x7FFFFFFFUs
//        = {9223372034707292159, 9223372034707292159};
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

static int exitFlag = 0;
#endif //DEMODULATOR_DEMODULATOR_H
