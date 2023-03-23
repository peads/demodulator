/*
 * This file is part of the demodulator distribution
 * (https://github.com/peads/demodulator).
 * and code originally part of the misc_snippets distribution
 * (https://github.com/peads/misc_snippets).
 * Copyright (c) 2023 Patrick Eads.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef DEMODULATOR_DEMODULATOR_H
#define DEMODULATOR_DEMODULATOR_H

#include <stdio.h>
#include <stdint.h>
#include <immintrin.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

//#define DEBUG

// sizeof(uint8_t)
#define INPUT_ELEMENT_BYTES 1
// sizeof(float) >> 1
#define OUTPUT_ELEMENT_BYTES 4
// sizeof(__m128)
#define MATRIX_ELEMENT_BYTES 16
#define MATRIX_WIDTH 4
#define DEFAULT_BUF_SIZE 1024

#define PERFORM_DESPIKE(AVG, RESULT, IDX) \
    AVG = _mm_add_ps(AVG, _mm_mul_ps(DC_RAW_CONST, _mm_sub_ps(RESULT[IDX], AVG))); \
    RESULT[IDX] = _mm_sub_ps(RESULT[IDX], AVG);

#define PERFORM_DEMOD(RESULT, BUF, IDX, JDX) \
    RESULT[JDX] = arg(BUF[IDX]); \
    RESULT[JDX + 1] = arg(_mm_blend_ps(BUF[IDX], BUF[IDX + 1], 0b0011));

struct rotationMatrix {
    const __m128 a1;
    const __m128 a2;
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

static const __m128 DC_RAW_CONST = {1e-05f, 1e-05f, 1e-05f, 1e-05f};
__attribute__((used)) static const __m256 NEGATE_B_IM = {1.f, 1.f, 1.f, -1.f, 1.f, 1.f, 1.f, -1.f};
__attribute__((used)) static const __m256 ALL_64S = {64.f, 64.f, 64.f, 64.f, 64.f, 64.f, 64.f, 64.f};
__attribute__((used)) static const __m256 ALL_41S = {41.f, 41.f, 41.f, 41.f, 41.f, 41.f, 41.f, 41.f};
__attribute__((used)) static const __m256 ALL_23S = {23.f, 23.f, 23.f, 23.f, 23.f, 23.f, 23.f, 23.f};
__attribute__((used)) static const __m128 ALL_HUNDREDTHS = {0.01f, 0.01f, 0.01f, 0.01f};
__attribute__((used)) static const __m128i Z = {-0x7f7f7f7f7f7f7f7f, -0x7f7f7f7f7f7f7f7f}; // all -127s
static const struct rotationMatrix CONJ_TRANSFORM = {
        {1, 0, 1, 0},
        {0, -1, 0, -1}
};

static int exitFlag = 0;
#endif //DEMODULATOR_DEMODULATOR_H
