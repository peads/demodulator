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
#include <immintrin.h>
#include "definitions.h"
#include "matrix.h"

typedef __m128 (*mm_convert_fun_t)(__m128i u);

typedef union {
    uint8_t arr[16];
    __m128i v;
} pun128u8;

typedef union {
    __m256 v;
    float arr[8];
} pun256f32;

static __m128 convertInt16ToFloat(__m128i u) {

    return _mm_cvtepi32_ps(
            _mm_cvtepi16_epi32(u));
}

static __m128 convertUint8ToFloat(__m128i u) {

    static const __m128i Z = {-0x7f7f7f7f7f7f7f7f, -0x7f7f7f7f7f7f7f7f};

    return convertInt16ToFloat(_mm_cvtepi8_epi16(_mm_add_epi16(u, Z)));
}

static inline __m128 conju(__m128 u) {

    static const __m128 Z = {1.f, -1.f, 1.f, -1.f};
    return _mm_mul_ps(u, Z);
}

static inline __m128 boxcar(__m128 u) {

    return _mm_add_ps(u, _mm_permute_ps(u, 0x4E));
}

static inline __m256 gather(__m128 u, __m128 v) {

    static const __m256 negateBIm = {1.f, 1.f, 1.f, -1.f, 1.f, 1.f, 1.f, -1.f};

    return _mm256_mul_ps(_mm256_set_m128(_mm_blend_ps(u, v, 0b0011), u), negateBIm);
}

static inline void preNormMult(__m256 *u, __m256 *v) {
    *v = _mm256_permute_ps(*u, 0xEB);                         //  {bj, br, br, bj, bj, br, br, bj} *
//    u = _mm256_permute_ps(u, 0x5);                        //  {aj, aj, ar, ar, cj, cj, cr, cr}
    *u = _mm256_mul_ps(_mm256_permute_ps(*u, 0x5), *v);  // = {aj*bj, aj*br, ar*br, ar*bj, bj*cj, br*cj, br*cr, bj*cr}
}

static inline void preNormAddSubAdd(__m256 *u, __m256 *v, __m256 *w) {
    *w = _mm256_permute_ps(*u, 0x8D);         // {aj, bj, ar, br, cj, dj, cr, dr}
    *u = _mm256_addsub_ps(*u, *w);     // {ar-aj, aj+bj, br-ar, bj+br, cr-cj, cj+dj, dr-cr, dj+dr}
    *v = _mm256_mul_ps(*u, *u);        // {(ar-aj)^2, (aj+bj)^2, (br-ar)^2, (bj+br)^2, (cr-cj)^2, (cj+dj)^2, (dr-cr)^2, (dj+dr)^2}
    *w = _mm256_permute_ps(*v, 0x1B);        // {ar^2, aj^2, br^2, bj^2, cr^2, cj^2, dr^2, dj^2} +
                                             // {bj^2, br^2, aj^2, ar^2, ... }
    *v = _mm256_add_ps(*v, *w);       // = {ar^2+bj^2, aj^2+br^2, br^2+aj^2, bj^2+ar^2, ... }
}

static float fmDemod(__m128 x) {

    static __m128 prev = {0.f, 0.f, 0.f, 0.f};

    static const __m256 all64s = {64.f, 64.f, 64.f, 64.f, 64.f, 64.f, 64.f, 64.f};
    static const __m256 all23s = {23.f, 23.f, 23.f, 23.f, 23.f, 23.f, 23.f, 23.f};
    static const __m256 all41s = {41.f, 41.f, 41.f, 41.f, 41.f, 41.f, 41.f, 41.f};

    pun256f32 ret;
    __m128 u0 = prev;
    __m128 v0 = prev = x;
    __m256 u, v, w, y;

    u = gather(u0, v0);
    preNormMult(&u, &v);
    preNormAddSubAdd(&u, &v, &w);

    v = _mm256_rsqrt_ps(v);
    u = _mm256_mul_ps(u, v);

    w = _mm256_mul_ps(u, all64s);                  // 64*zj
    u = _mm256_fmadd_ps(all23s, u, all41s);     // 23*zr + 41s
    y = _mm256_rcp_ps(_mm256_permute_ps(u, 0x1B));
    u = _mm256_mul_ps(w, y);

    v = _mm256_cmp_ps(u, u, 0);                           // NAN check
    u = _mm256_and_ps(u, v);

    ret.v = u;
    return ret.arr[5];
}

static inline mm_convert_fun_t processMode(const uint8_t mode) {

    switch (mode) {
        case 0: // default mode (input int16)
            return convertInt16ToFloat;
        case 1: // input uint8
            return convertUint8ToFloat;
        default:
            return NULL;
    }
}

int processMatrix(FILE *__restrict__ inFile,
                  uint8_t mode,
                  float gain,
                  void *__restrict__ outFile) {

    static pun128u8 x;
    static float result[DEFAULT_BUF_SIZE >> 2];

    const size_t inputElementBytes = 2 - mode;
    const uint8_t isGain = fabsf(1.f - gain) > GAIN_THRESHOLD;
    const mm_convert_fun_t convert = processMode(mode);

    int exitFlag = mode < 0 || mode > 2;
    size_t readBytes, i;
    float ret;
    // TODO change/add pre-demodulation gain, s.t. we leverage simd
    while (!exitFlag) {
        for (i = 0, readBytes = 0; i < DEFAULT_BUF_SIZE; i += OUTPUT_ELEMENT_BYTES) {

            readBytes += fread(x.arr, inputElementBytes, OUTPUT_ELEMENT_BYTES, inFile);

            if ((exitFlag = ferror(inFile))) {
                perror(NULL);
                break;
            } else if (feof(inFile)) {
                exitFlag = EOF;
            }

            ret = fmDemod(boxcar(conju(convert(x.v))));
            result[i >> 2] = isGain ? ret * gain : ret;
        }

        fwrite(result, OUTPUT_ELEMENT_BYTES, readBytes >> 2, outFile);
    }

    return exitFlag;
}