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

__attribute__((used)) static const __m128 CNJ_TRANSFORM = {1.f, -1.f, 1.f, -1.f};
__attribute__((used)) static const __m128 DC_RAW_CONST = {1e-05f, 1e-05f, 1e-05f, 1e-05f};
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

__attribute__((used)) static __m128 dc_avg_iq = {0,0,0,0};
static int exitFlag = 0;

/**
 * Takes a 4x4 matrix and applies it to a 4x1 vector.
 * Here, it is used to apply the same rotation matrix to
 * two complex numbers. i.e., for the the matrix
 * T = {{a,b}, {c,d}} and two vectors {u1,u2} and {v1,v2}
 * concatenated, s.t. u = {u1,u2,v1,v2}, Tu =
 * {a*u1 + c*u1, b*u2 + d*u2, ... , b*v2 + d*v2}
 */
__asm__(
#ifdef __clang__
"_apply4x4_4x1Transform: "
#else
"apply4x4_4x1Transform: "
#endif
    "vmulps 16(%rdi), %xmm0, %xmm2\n\t"      // u1*a11, u2*a12, u3*a13, ...
    "vmulps (%rdi), %xmm0, %xmm1\n\t"        // u1*a21, u2*a22, ...
    "vpermilps $0xB1, %xmm2, %xmm0\n\t"
    "vaddps %xmm2, %xmm0, %xmm2\n\t"         // u1*a11 + u2*a12, ... , u3*a13 + u4*a14
    "vpermilps $0xB1, %xmm1, %xmm0\n\t"
    "vaddps %xmm1, %xmm0, %xmm1\n\t"         // u1*a21 + u2*a22, ... , u3*a23 + u4*a24
    "vblendps $0xA, %xmm2, %xmm1, %xmm0\n\t" // u1*a11 + u2*a12, u1*a21 + u2*a22,
    "ret"                                    // u3*a13 + u4*a14, u3*a23 + u4*a24
);
#endif //DEMODULATOR_DEMODULATOR_H
