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

#ifdef __GNUC__

#include <stdint.h>

#endif

#include <immintrin.h>
#include <math.h>
#include "definitions.h"
#include "matrix.h"

typedef union {
    __m256i v;
    uint8_t uint8Arr[32];
    int16_t int16Arr[16];
} pun256Int;

typedef union {
    __m256 v;
    float arr[8];
} pun256f32;

typedef union {
    float arr[4];
    __m128 v;
} pun128f32;

static __m128 convertInt16ToFloat(__m128i u) {

    return _mm_cvtepi32_ps(_mm_cvtepi16_epi32(u));
}

static inline __m256i convertUint8ToInt8(__m256i u) {

    static const __m256i Z = {-0x7f7f7f7f7f7f7f7f, -0x7f7f7f7f7f7f7f7f, -0x7f7f7f7f7f7f7f7f, -0x7f7f7f7f7f7f7f7f};
    return _mm256_add_epi8(u, Z);
}

static __m128 convertInt8ToFloat(__m128i u) {

    return convertInt16ToFloat(_mm_cvtepi8_epi16(u));
}

static inline __m256i conju(__m256i u) {

    static const __m256i mask = {
            0x0606040402020000, 0x0e0e0c0c0a0a0808,
            0x0606040402020000, 0x0e0e0c0c0a0a0808};
    static const __m256i mask2 = {
            (int64_t) 0x8000800080008000, (int64_t) 0x8000800080008000,
            (int64_t) 0x8000800080008000, (int64_t) 0x8000800080008000};
    static const __m256i Z = {
            (int64_t) 0xff01ff01ff01ff01,
            (int64_t) 0xff01ff01ff01ff01,
            (int64_t) 0xff01ff01ff01ff01,
            (int64_t) 0xff01ff01ff01ff01};

    __m256i v = u;
    __asm__(
            "VPMULHW %0, %1, %0\n\t"
            : "+x" (v): "x" (Z)
            );

    return _mm256_blendv_epi8(u, _mm256_shuffle_epi8(v, mask), mask2);
}

static inline __m256i boxcar(__m256i u) {

    static const __m256i mask = {
            0x0504070601000302, 0x0d0c0f0e09080b0a,
            0x0504070601000302, 0x0d0c0f0e09080b0a};
    return _mm256_add_epi8(u, _mm256_shuffle_epi8(u, mask));
}

static inline __m256 gather(__m128 u, __m128 v) {

    static const __m256 negateBIm = {1.f, 1.f, 1.f, -1.f, 1.f, 1.f, 1.f, -1.f}; //0x010101FF...

    return _mm256_mul_ps(_mm256_set_m128(_mm_blend_ps(u, v, 0b0011), u), negateBIm);
}

static inline void preMultNorm(__m256 *u, __m256 *v) {

    *v = _mm256_permute_ps(*u, 0xEB);   //  {bj, br, br, bj, bj, br, br, bj} *
                                        //  {aj, aj, ar, ar, cj, cj, cr, cr}
                                        // = {aj*bj, aj*br, ar*br, ar*bj, bj*cj, br*cj, br*cr, bj*cr}
    *u = _mm256_mul_ps(_mm256_permute_ps(*u, 0x5),*v);
}

static inline void foo(__m256 *u, __m256 *v, __m256 *w) {

    *w = _mm256_permute_ps(*u, 0x8D);         // {aj, bj, ar, br, cj, dj, cr, dr}
    *u = _mm256_addsub_ps(*u, *w);     // {ar-aj, aj+bj, br-ar, bj+br, cr-cj, cj+dj, dr-cr, dj+dr}
    *v = _mm256_mul_ps(*u,
            *u);        // {(ar-aj)^2, (aj+bj)^2, (br-ar)^2, (bj+br)^2, (cr-cj)^2, (cj+dj)^2, (dr-cr)^2, (dj+dr)^2}
    *w = _mm256_permute_ps(*v, 0x1B);        // {ar^2, aj^2, br^2, bj^2, cr^2, cj^2, dr^2, dj^2} +
    // {bj^2, br^2, aj^2, ar^2, ... }
    *v = _mm256_add_ps(*v, *w);       // = {ar^2+bj^2, aj^2+br^2, br^2+aj^2, bj^2+ar^2, ... }
}

static float fmDemod(__m128 x) {

    static const __m256 all64s = {64.f, 64.f, 64.f, 64.f, 64.f, 64.f, 64.f, 64.f};
    static const __m256 all23s = {23.f, 23.f, 23.f, 23.f, 23.f, 23.f, 23.f, 23.f};
    static const __m256 all41s = {41.f, 41.f, 41.f, 41.f, 41.f, 41.f, 41.f, 41.f};

    static __m128 prev = {0.f, 0.f, 0.f, 0.f};

    pun256f32 ret;
    __m128 u0 = prev;
    __m128 v0 = prev = x;
    __m256 v, w, y;
    __m256 u = gather(u0, v0); // {ar, aj, br, bj, cr, cj, br, bj}

    preMultNorm(&u, &v);
    foo(&u, &v, &w);

    v = _mm256_rsqrt_ps(v);
    u = _mm256_mul_ps(u, v);

    w = _mm256_mul_ps(u, all64s);                  // 64*zj
    u = _mm256_fmadd_ps(all23s, u, all41s);     // 23*zr + 41s
    y = _mm256_rcp_ps(_mm256_permute_ps(u, 0x1B));
    u = _mm256_mul_ps(w, y);

    v = _mm256_cmp_ps(u, u, 0);                           // NAN check
    ret.v = _mm256_and_ps(u, v);

    return ret.arr[5];
}

int processMatrix(FILE *__restrict__ inFile, uint8_t mode, const float inGain,
                  void *__restrict__ outFile) {

    int exitFlag = mode < 0 || mode > 2;
    size_t readBytes = 0;
    pun128f32 resultVector = {};
    float *result = resultVector.arr;
    __m128i lo, hi;
    __m256i v;
    pun256Int z;

    const size_t inputElementBytes = 1;//2 - mode; // TODO
    const uint8_t isGain = fabsf(1.f - inGain) > GAIN_THRESHOLD;
    const __m128 gain = _mm_broadcast_ss(&inGain);

    while (!exitFlag) {

        readBytes += fread(z.uint8Arr, inputElementBytes, 32, inFile);

        if ((exitFlag = ferror(inFile))) {
            perror(NULL);
            break;
        } else if (feof(inFile)) {
            exitFlag = EOF;
        }

        v = boxcar(conju(convertUint8ToInt8(z.v)));
        lo = _mm256_castsi256_si128(v);
        hi = _mm256_extracti128_si256(v, 1);

        result[0] = fmDemod(convertInt8ToFloat(_mm_movpi64_epi64(*(__m64 *) (&lo))));
        result[1] = fmDemod(convertInt8ToFloat(_mm_movpi64_epi64((__m64) _mm_extract_epi64(lo,
                1))));
        result[2] = fmDemod(convertInt8ToFloat(_mm_movpi64_epi64(*(__m64 *) (&hi))));
        result[3] = fmDemod(convertInt8ToFloat(_mm_movpi64_epi64((__m64) _mm_extract_epi64(hi,
                1))));

        if (isGain) {
            _mm_mul_ps(resultVector.v, gain);
        }

        fwrite(result, OUTPUT_ELEMENT_BYTES, MATRIX_WIDTH, outFile);
    }

    printf("Total bytes read: %lu\n", readBytes);
    return exitFlag;
}
