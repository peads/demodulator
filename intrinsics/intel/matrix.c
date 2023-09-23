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
#ifdef __INTEL_COMPILER
#include <stdlib.h>
#endif

#include <immintrin.h>
#include <math.h>
#include "definitions.h"
#include "matrix.h"

typedef union {
    __m256i v;
    int8_t buf[32];
    int16_t buf16[16];
} m256i_pun_t;

typedef void (*matrixOp256_t)(__m256i, float *__restrict__);

static inline void convert_epi16_epi32(__m256i *__restrict__ u, __m256i *__restrict__ v) {

    *v = _mm256_cvtepi16_epi32(_mm256_extractf128_si256(*u, 1));
    *u = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(*u));
}

static inline void convert_epi16_ps(__m256i u, __m256 *__restrict__ ret) {

    __m256i q1;

    convert_epi16_epi32(&u, &q1);
    ret[0] = _mm256_cvtepi32_ps(u);
    ret[1] = _mm256_cvtepi32_ps(q1);
}

static inline __m256i convert_epu8_epi8(__m256i u) {

    static const __m256i Z = {
            -0x7f7f7f7f7f7f7f7f,
            -0x7f7f7f7f7f7f7f7f,
            -0x7f7f7f7f7f7f7f7f,
            -0x7f7f7f7f7f7f7f7f};

    return _mm256_add_epi8(u, Z);
}

static inline void convert_epi8_epi16(__m256i *__restrict__ u, __m256i *__restrict__ v) {

    *v = _mm256_cvtepi8_epi16(_mm256_extractf128_si256(*u, 1));
    *u = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(*u));
}

static inline void convert_epi8_ps(__m256i u, __m256 *__restrict__ ret) {

    __m256i v = {};
    convert_epi8_epi16(&u, &v);
    convert_epi16_ps(u, ret);
    convert_epi16_ps(v, &(ret[2]));
}

static inline __m256i boxcarEpi8(__m256i u) {

    static const __m256i Z = {
            (int64_t) 0xff01ff01ff01ff01,
            (int64_t) 0xff01ff01ff01ff01,
            (int64_t) 0xff01ff01ff01ff01,
            (int64_t) 0xff01ff01ff01ff01};
    static const __m256i mask = {
            0x0504070601000302, 0x0d0c0f0e09080b0a,
            0x0504070601000302, 0x0d0c0f0e09080b0a};

    u = _mm256_sign_epi8(u, Z);
    return _mm256_add_epi8(u, _mm256_shuffle_epi8(u, mask));
}

static inline __m256i boxcarEpi16(__m256i u) {

    static const __m256i Z = {
        (int64_t) 0xffff0001ffff0001,
        (int64_t) 0xffff0001ffff0001,
        (int64_t) 0xffff0001ffff0001,
        (int64_t) 0xffff0001ffff0001};
    static const __m256i mask = {
        0x0302010007060504, 0x0b0a09080f0e0d0c,
        0x0302010007060504, 0x0b0a09080f0e0d0c};
    u = _mm256_sign_epi16(u, Z);
    return _mm256_add_epi16(u, _mm256_shuffle_epi8(u, mask));
}

static inline void preNormMult(__m256 *u, __m256 *v) {

    *v = _mm256_permute_ps(*u, 0xEB);   //  {bj, br, br, bj, bj, br, br, bj} *
                                        //  {aj, aj, ar, ar, cj, cj, cr, cr}
                                        // = {aj*bj, aj*br, ar*br, ar*bj, bj*cj, br*cj, br*cr, bj*cr}
    *u = _mm256_mul_ps(_mm256_permute_ps(*u, 0x5), *v);
}

static inline void preNormAddSubAdd(__m256 *u, __m256 *v, __m256 *w) {

    *w = _mm256_permute_ps(*u, 0x8D);         // {aj, bj, ar, br, cj, dj, cr, dr}
    *u = _mm256_addsub_ps(*u, *w);     // {ar-aj, aj+bj, br-ar, bj+br, cr-cj, cj+dj, dr-cr, dj+dr}
    *v = _mm256_mul_ps(*u,*u);         // {(ar-aj)^2, (aj+bj)^2, (br-ar)^2, (bj+br)^2, (cr-cj)^2, (cj+dj)^2, (dr-cr)^2, (dj+dr)^2}
    *w = _mm256_permute_ps(*v, 0x1B);        // {ar^2, aj^2, br^2, bj^2, cr^2, cj^2, dr^2, dj^2} +
                                             // {bj^2, br^2, aj^2, ar^2, ... }
    *v = _mm256_add_ps(*v, *w);       // = {ar^2+bj^2, aj^2+br^2, br^2+aj^2, bj^2+ar^2, ... }
}

static float fmDemod(__m256 u, __m256 v, __m256 w) {

    static const __m256 all64s = {64.f, 64.f, 64.f, 64.f, 64.f, 64.f, 64.f, 64.f};
    static const __m256 all23s = {23.f, 23.f, 23.f, 23.f, 23.f, 23.f, 23.f, 23.f};
    static const __m256 all41s = {41.f, 41.f, 41.f, 41.f, 41.f, 41.f, 41.f, 41.f};

    __m256 y;

    // Norm
    v = _mm256_rsqrt_ps(v);
    u = _mm256_mul_ps(u, v);

    // fast atan2 -> atan2(x,y) = 64y/(23x+41)
    w = _mm256_mul_ps(u, all64s);                  // 64*zj
    u = _mm256_fmadd_ps(all23s, u, all41s);     // 23*zr + 41s
    y = _mm256_rcp_ps(_mm256_permute_ps(u, 0x1B));
    u = _mm256_mul_ps(w, y);

    v = _mm256_cmp_ps(u, u, 0);                           // NAN check
    u = _mm256_and_ps(u, v);

    return u[5];
}

static inline void demod(__m256 *__restrict__ M, float *__restrict__ result) {

    preNormMult(M, &(M[2]));
    preNormMult(&(M[1]), &(M[3]));

    preNormAddSubAdd(&M[0], &M[2], &M[4]);
    preNormAddSubAdd(&M[1], &M[3], &M[5]);

    result[0] = fmDemod(M[0], M[2], M[4]);
    result[1] = fmDemod(M[1], M[3], M[5]);
}

static inline void demodEpi16(__m256i u, float *__restrict__ result) {

    static const __m256i negateBIm = {
        (int64_t) 0xffff000100010001,
        (int64_t) 0xffff000100010001,
        (int64_t) 0xffff000100010001,
        (int64_t) 0xffff000100010001};

    static m256i_pun_t prev;

    __m256i hi;
    m256i_pun_t lo;

    __m256 M[6];

    u = boxcarEpi16(u);

    const __m256i indexLo = _mm256_setr_epi16(
        0,0,1,1,2,2,1,1,
        2,2,3,3,4,4,3,3
    );
    const __m256i indexHi = _mm256_setr_epi16(
        4,4,5,5,6,6,5,5,
        6,6,7,7,8,8,7,7
    );

    hi = _mm256_sign_epi16(_mm256_permutevar8x32_epi32(u, indexHi), negateBIm);
    lo.v = _mm256_sign_epi16(_mm256_permutevar8x32_epi32(u, indexLo), negateBIm);

    prev.buf16[12] = lo.buf16[0];
    prev.buf16[13] = lo.buf16[1];

    convert_epi16_ps(prev.v, M);
    demod(M, result);

    convert_epi16_ps(lo.v, M);
    demod(M, &(result[2]));

    prev.v = hi;
}

static inline void demodEpi8(__m256i u, float *__restrict__ result) {

    static const __m256i negateBIm = {
        (int64_t) 0xff010101ff010101,
        (int64_t) 0xff010101ff010101,
        (int64_t) 0xff010101ff010101,
        (int64_t) 0xff010101ff010101};

    static const __m256i indexHi = {
        0x1312151413121110,
        0x1716191815161514,
        0x1b1a1d1c1b1a1918,
        0x1f1e21201f1e1d1c};

    static const __m256i indexLo = {
        0x302050403020100,
        0x706090807060504,
        0xb0a0d0c0b0a0908,
        0xf0e11100f0e0d0c};

    static m256i_pun_t prev;

    __m256i hi;
    m256i_pun_t lo;

    __m256 M[6];
    __m256 temp[2];

    u = boxcarEpi8(convert_epu8_epi8(u));


    hi = _mm256_sign_epi8(_mm256_permutevar8x32_epi32(u, indexHi), negateBIm);
    lo.v = _mm256_sign_epi8(_mm256_permutevar8x32_epi32(u, indexLo), negateBIm);

    prev.buf[28] = lo.buf[0];
    prev.buf[29] = lo.buf[1];

    convert_epi8_ps(prev.v, M);
    temp[0] = M[2];
    temp[1] = M[3];

    demod(M, result);
    M[0] = temp[0];
    M[1] = temp[1];
    demod(M, &(result[2]));

    convert_epi8_ps(lo.v, M);
    temp[0] = M[2];
    temp[1] = M[3];

    demod(M, &(result[4]));
    M[0] = temp[0];
    M[1] = temp[1];
    demod(M, &(result[6]));

    prev.v = hi;
}

int processMatrix(FILE *__restrict__ inFile,
                  const uint8_t mode,
                  const float inGain,
                  void *__restrict__ outFile) {

    int exitFlag = 0;
    size_t elementsRead;
    void *buf = _mm_malloc(MATRIX_WIDTH << 3, 32);
    float result[MATRIX_WIDTH << 1];

    const uint8_t isGain = inGain != 1.f && fabsf(inGain) > GAIN_THRESHOLD;
    const __m256 gain = _mm256_broadcast_ss(&inGain);
    const matrixOp256_t demodulate = mode ? demodEpi8 : demodEpi16;


    while (!exitFlag) {

        elementsRead = fread(buf, 1, MATRIX_WIDTH << 3, inFile);

        if ((exitFlag = ferror(inFile))) {
            perror(NULL);
            break;
        } else if (feof(inFile)) {
            exitFlag = EOF;
        } else if (!elementsRead) {
            fprintf(stderr, "This shouldn't happen, but I need to use the result of"
                            "fread. Stupid compiler.");
        }

        demodulate(*(__m256i *) buf, result);

        if (isGain) {
            _mm256_mul_ps(*(__m256 *) &result, gain);
        }

        fwrite(result, OUTPUT_ELEMENT_BYTES, MATRIX_WIDTH >> (1-mode), outFile);
    }

    _mm_free(buf);
    return exitFlag;
}
