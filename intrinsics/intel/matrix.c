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

#ifdef __INTEL_COMPILER
#include <stdlib.h>
#endif

#include <immintrin.h>
#include "matrix.h"

typedef union {
    __m256i v;
    int8_t buf[32];
} m256i_pun_t;

static inline __m256i convert_epu8_epi8(__m256i u) {

    static const __m256i Z = {
            -0x7f7f7f7f7f7f7f7f,
            -0x7f7f7f7f7f7f7f7f,
            -0x7f7f7f7f7f7f7f7f,
            -0x7f7f7f7f7f7f7f7f};

    return _mm256_add_epi8(u, Z);
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

static inline void preNormMult(__m256 *u, __m256 *v) {

    *v = _mm256_permute_ps(*u, 0xEB);
    *u = _mm256_mul_ps(_mm256_permute_ps(*u, 0x5), *v);
}

static inline void preNormAddSubAdd(__m256 *u, __m256 *v, __m256 *w) {

    *w = _mm256_permute_ps(*u, 0x8D);
    *u = _mm256_addsub_ps(*u, *w);
    *v = _mm256_mul_ps(*u, *u);
    *w = _mm256_permute_ps(*v, 0x1B);
    *v = _mm256_add_ps(*v, *w);
}

static float fmDemod(__m256 *M) {

    static const __m256 all64s = {64.f, 64.f, 64.f, 64.f, 64.f, 64.f, 64.f, 64.f};
    static const __m256 all23s = {23.f, 23.f, 23.f, 23.f, 23.f, 23.f, 23.f, 23.f};
    static const __m256 all41s = {41.f, 41.f, 41.f, 41.f, 41.f, 41.f, 41.f, 41.f};

    __m256 w, y,
            u = M[0],
            v = M[2];

    // Norm
    preNormMult(&u, &w);
    preNormAddSubAdd(&u, &v, &w);
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

static inline void convert_epi8_ps(__m256i u, __m256 *__restrict__ ret) {

    __m256i w,
            v = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(u, 1));
    u = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(u));

    w = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(u, 1));
    u = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(u));

    ret[0] = _mm256_cvtepi32_ps(u);
    ret[1] = _mm256_cvtepi32_ps(w);

    w = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(v, 1));
    v = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v));

    ret[2] = _mm256_cvtepi32_ps(v);
    ret[3] = _mm256_cvtepi32_ps(w);
}

static inline void demod(__m256 *__restrict__ M, float *__restrict__ result) {

    result[0] = fmDemod(M);
    result[1] = fmDemod(&M[1]);
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

    __m256 M[4];

    u = boxcarEpi8(convert_epu8_epi8(u));

    hi = _mm256_sign_epi8(_mm256_permutevar8x32_epi32(u, indexHi), negateBIm);
    lo.v = _mm256_sign_epi8(_mm256_permutevar8x32_epi32(u, indexLo), negateBIm);

    prev.buf[28] = lo.buf[0];
    prev.buf[29] = lo.buf[1];

    convert_epi8_ps(prev.v, M);
    demod(M, result);
    demod(&M[2], result);

    convert_epi8_ps(lo.v, M);
    demod(M, &result[2]);
    demod(&M[2], &result[2]);

    prev.v = hi;
}

void *processMatrix(void *ctx) {

    consumerArgs *args = ctx;
    size_t i, j;
    void *buf = _mm_malloc(DEFAULT_BUF_SIZE, 32);
    float result[DEFAULT_BUF_SIZE >> 3];
    __m256 gain = _mm256_broadcast_ss(&args->gain);

    while (!args->exitFlag) {
        sem_wait(&args->full);
        pthread_mutex_lock(&args->mutex);
        memcpy(buf, args->buf, DEFAULT_BUF_SIZE);
        pthread_mutex_unlock(&args->mutex);
        sem_post(&args->empty);

        for (i = 0, j = 0; i < DEFAULT_BUF_SIZE; i += 32, j += 4) {
            demodEpi8(*(__m256i *) (buf + i), result + j);

            if (*(float *) &args->gain) {
                _mm256_mul_ps(*(__m256 *) &result, gain);
            }
        }
        fwrite(result, sizeof(float), DEFAULT_BUF_SIZE >> 3, args->outFile);
    }

    _mm_free(buf);
    return NULL;
}

void allocateBuffer(void **buf, const size_t len) {
    *buf = _mm_malloc(len, 64);
}
