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
size_t iterations = 0;
typedef union {
    __m256i v;
    int8_t buf[32];
} m256i_pun_t;

static inline __m256i shiftOrigin(__m256i u) {

    static const __m256i shift = {
            -0x7f7f7f7f7f7f7f7f,
            -0x7f7f7f7f7f7f7f7f,
            -0x7f7f7f7f7f7f7f7f,
            -0x7f7f7f7f7f7f7f7f};

    return _mm256_add_epi8(u, shift);
}

static inline void convert_epi8_epi16(__m256i u, __m256i *hi, __m256i *lo) {

    *hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(u, 1));
    *lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(u));
}

static inline void convert_epi16_ps(__m256i u, __m256 *__restrict__ ret) {

    __m256i w;

    w = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(u, 1));
    u = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(u));

    ret[0] = _mm256_cvtepi32_ps(u);
    ret[1] = _mm256_cvtepi32_ps(w);
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

static inline void complexMultiply(__m256i u, __m256i *ulo, __m256i *uhi) {

    const __m256i indexHiSymmetry = _mm256_setr_epi8(
            3, 2, 2, 3, 7, 6, 6, 7,
            11, 10, 10, 11, 15, 14, 14, 15,
            19, 18, 18, 19, 23, 22, 22, 23,
            27, 26, 26, 27, 31, 30, 30, 31
    );

    const __m256i indexLoDuplicateReverse = _mm256_setr_epi8(
            0, 1, 0, 1, 4, 5, 4, 5,
            8, 9, 8, 9, 12, 13, 12, 13,
            16, 17, 16, 17, 20, 21, 20, 21,
            24, 25, 24, 25, 28, 29, 28, 29
    );

    const __m256i negs = _mm256_setr_epi16(
            -1, 1, -1, 1, -1, 1, -1, 1,
            -1, 1, -1, 1, -1, 1, -1, 1);

    __m256i vlo, vhi,
    v = _mm256_shuffle_epi8(u, indexHiSymmetry);
    u = _mm256_shuffle_epi8(u, indexLoDuplicateReverse);
    convert_epi8_epi16(u, uhi, ulo);
    convert_epi8_epi16(v, &vhi, &vlo);
    *ulo = _mm256_mullo_epi16(*ulo, vlo);
    *uhi = _mm256_mullo_epi16(*uhi, vhi);

    vlo = _mm256_shufflelo_epi16(_mm256_shufflehi_epi16(*ulo, SHUF_INDEX), SHUF_INDEX); // 2031_4
    *ulo = _mm256_add_epi16(*ulo, _mm256_sign_epi16(vlo, negs));
    vhi = _mm256_shufflelo_epi16(_mm256_shufflehi_epi16(*uhi, SHUF_INDEX), SHUF_INDEX); // 2031_4
    *uhi = _mm256_add_epi16(*uhi, _mm256_sign_epi16(vhi, negs));
}

static float fmDemod(__m256 u) {

    static const __m256 all64s = {64.f, 64.f, 64.f, 64.f, 64.f, 64.f, 64.f, 64.f};
    static const __m256 all23s = {23.f, 23.f, 23.f, 23.f, 23.f, 23.f, 23.f, 23.f};
    static const __m256 all41s = {41.f, 41.f, 41.f, 41.f, 41.f, 41.f, 41.f, 41.f};

    __m256 w;

    // norm
    w = _mm256_mul_ps(u, u);
    w = _mm256_rsqrt_ps(_mm256_add_ps(w, _mm256_permute_ps(w, 0x1B)));
    u = _mm256_mul_ps(u, w);

    // fast atan2 -> atan2(x,y) = 64y/(23x+41)
    w = _mm256_mul_ps(u, all64s);                  // 64*zj
    u = _mm256_fmadd_ps(all23s, u, all41s);     // 23*zr + 41s
    u = _mm256_mul_ps(w, _mm256_rcp_ps(_mm256_permute_ps(u, 0x1B)));

    w = _mm256_cmp_ps(u, u, 0);                           // NAN check
    u = _mm256_and_ps(u, w);
    iterations++;
    return u[5];
}

static inline float demodEpi8(__m256i u) {

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

    float result[8];
    __m256i hi, uhi, ulo;
    m256i_pun_t lo;
    __m256 U[2];

    u = boxcarEpi8(shiftOrigin(u));

    hi = _mm256_sign_epi8(_mm256_permutevar8x32_epi32(u, indexHi), negateBIm);
    lo.v = _mm256_sign_epi8(_mm256_permutevar8x32_epi32(u, indexLo), negateBIm);

    prev.buf[28] = lo.buf[0];
    prev.buf[29] = lo.buf[1];

    complexMultiply(prev.v, &ulo, &uhi);
    convert_epi16_ps(ulo, U);
    result[0] =fmDemod(U[0]);
    result[1] =fmDemod(U[1]);
    convert_epi16_ps(uhi, U);
    result[2] =fmDemod(U[0]);
    result[3] = fmDemod(U[1]);

    complexMultiply(lo.v, &ulo, &uhi);
    convert_epi16_ps(ulo, U);
    result[4] =fmDemod(U[0]);
    result[5] =fmDemod(U[1]);
    convert_epi16_ps(uhi, U);
    result[6] =fmDemod(U[0]);
    result[7] = fmDemod(U[1]);

    prev.v = hi;
    return result[3];
}

void *processMatrix(void *ctx) {

    consumerArgs *args = ctx;
    size_t i;
    void *buf = _mm_malloc(DEFAULT_BUF_SIZE, ALIGNMENT);
    float result[DEFAULT_BUF_SIZE >> 5];
    __m256 gain = _mm256_broadcast_ss(&args->gain);

    while (!args->exitFlag) {
        sem_wait(&args->full);
        pthread_mutex_lock(&args->mutex);
        memcpy(buf, args->buf, DEFAULT_BUF_SIZE);
        pthread_mutex_unlock(&args->mutex);
        sem_post(&args->empty);

        for (i = 0; i < DEFAULT_BUF_SIZE; i += 32) {
            result[i >> 5] = demodEpi8(*(__m256i *) (buf + i));
        }

        if (*(float *) &args->gain) {
            _mm256_mul_ps(*(__m256 *) &result, gain);
        }
        fwrite(result, sizeof(float), DEFAULT_BUF_SIZE >> 5, args->outFile);
    }

    _mm_free(buf);
    return NULL;
}

void allocateBuffer(void **buf, const size_t len) {

    *buf = _mm_malloc(len, 32);
}
