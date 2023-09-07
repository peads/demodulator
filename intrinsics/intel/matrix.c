/*
 * This file is part of the demodulator distribution
 * (https://github.com/peads/demodulator).
 * with code originally part of the misc_snippets distribution
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
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <immintrin.h>
#include "definitions.h"
#include "matrix.h"

static inline __m128 convertToFloats(__m128i u) {

    static const __m128i Z = {-0x7f7f7f7f7f7f7f7f, -0x7f7f7f7f7f7f7f7f};

    return _mm_cvtepi32_ps(
        _mm_cvtepi16_epi32(_mm_cvtepi8_epi16(_mm_add_epi16(u, Z))));
}

static inline __m128 conju(__m128 u) {

    static const __m128 Z = {1.f, -1.f, 1.f, -1.f};
    return _mm_mul_ps(u, Z);
}

static inline __m128 boxcar(__m128 u) {

    return _mm_add_ps(u, _mm_permute_ps(u, 0x4E));
}

static float fmDemod(__m128 x) {

    static union {
        __m256 v;
        float arr[8];
    } ret;

    static const __m256 negateBIm = {1.f, 1.f, 1.f, -1.f, 1.f, 1.f, 1.f, -1.f};
    static const __m256 all64s = {64.f, 64.f, 64.f, 64.f, 64.f, 64.f, 64.f, 64.f};
    static const __m256 all23s = {23.f, 23.f, 23.f, 23.f, 23.f, 23.f, 23.f, 23.f};
    static const __m256 all41s = {41.f, 41.f, 41.f, 41.f, 41.f, 41.f, 41.f, 41.f};

    static __m128 prev = {0.f, 0.f, 0.f, 0.f};

    __m128 u0 = prev;
    __m128 v0 = prev = x;
    __m256 u, v, w, y;

    v0 = _mm_blend_ps(u0, v0, 0b0011);
    u = _mm256_set_m128(v0, u0);
    u = _mm256_mul_ps(u, negateBIm);

    v = _mm256_permute_ps(u, 0xEB);
    u = _mm256_permute_ps(u, 0x5);

    u = _mm256_mul_ps(u, v);
    w = _mm256_permute_ps(u, 0x8D);
    u = _mm256_addsub_ps(u, w);
    v = _mm256_mul_ps(u, u);
    w = _mm256_permute_ps(v, 0x1B);
    v = _mm256_add_ps(v, w);

    v = _mm256_rsqrt_ps(v);
    u = _mm256_mul_ps(u, v);

    w = _mm256_mul_ps(u, all64s);// 64*zj
    u = _mm256_fmadd_ps(all23s, u, all41s);// 23*zr + 41s
    y = _mm256_rcp_ps(_mm256_permute_ps(u, 0x1B));
    u = _mm256_mul_ps(w, y);
    // TODO NAN check
    ret.v = u;
    return ret.arr[5];
}

int processMatrix(FILE *__restrict__ inFile,
                  uint8_t mode,
                  float gain,
                  void *__restrict__ outFile) {

    static union {
        uint8_t arr[16];
        __m128i v;
    } x;

    const size_t inputElementBytes = 2 - mode;
//    const uint8_t isGain = fabsf(1.f - gain) > GAIN_THRESHOLD;

    int exitFlag = mode <= 0 || mode >= 3;

    size_t readBytes, i;
    float ret;
    float result[DEFAULT_BUF_SIZE >> 2];

    while (!exitFlag) {
        for (i = 0, readBytes = 0; i < DEFAULT_BUF_SIZE; i+=4) {

            readBytes += fread(x.arr, inputElementBytes, 4, inFile);

            if ((exitFlag = ferror(inFile))) {
                perror(NULL);
                break;
            } else if (feof(inFile)) {
                exitFlag = EOF;
            }

            ret = fmDemod(boxcar(conju(convertToFloats(x.v))));
            result[i>>2] = ret;
        }

        fwrite(result, OUTPUT_ELEMENT_BYTES, readBytes>>2, outFile);
    }

    return exitFlag;
}