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
    __m512i v;
    int8_t buf[64];
} m512i_pun_t;

// taken from https://stackoverflow.com/a/55745816
static inline __m512i conditional_negate_epi16(__m512i target, __m512i signs) {

    static const __m512i ZEROS = {};
    // vpsubb target{k1}, 0, target
    return _mm512_mask_sub_epi16(target, _mm512_movepi16_mask(signs), ZEROS, target);
}

// taken from https://stackoverflow.com/a/55745816
static inline __m512i conditional_negate_epi8(__m512i target, __m512i signs) {

    static const __m512i ZEROS = {};
    // vpsubb target{k1}, 0, target
    return _mm512_mask_sub_epi8(target, _mm512_movepi8_mask(signs), ZEROS, target);
}

static inline __m512i shiftOrigin(__m512i u) {

    static const __m512i Z = {
            -0x7f7f7f7f7f7f7f7f,
            -0x7f7f7f7f7f7f7f7f,
            -0x7f7f7f7f7f7f7f7f,
            -0x7f7f7f7f7f7f7f7f,
            -0x7f7f7f7f7f7f7f7f,
            -0x7f7f7f7f7f7f7f7f,
            -0x7f7f7f7f7f7f7f7f,
            -0x7f7f7f7f7f7f7f7f};

    return _mm512_add_epi8(u, Z);
}

static inline void convert_epi8_epi16(__m512i u, __m512i *hi, __m512i *lo) {

    *hi = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(u, 1));
    *lo = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(u));
}

static inline void convert_epi16_ps(__m512i u, __m512 *__restrict__ ret) {

    __m512i w;

    w = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(u, 1));
    u = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u));

    ret[0] = _mm512_cvtepi32_ps(u);
    ret[1] = _mm512_cvtepi32_ps(w);
}

static inline __m512i boxcarEpi8(__m512i u) {

    static const __m512i Z = {
            (int64_t) 0xff01ff01ff01ff01,
            (int64_t) 0xff01ff01ff01ff01,
            (int64_t) 0xff01ff01ff01ff01,
            (int64_t) 0xff01ff01ff01ff01,
            (int64_t) 0xff01ff01ff01ff01,
            (int64_t) 0xff01ff01ff01ff01,
            (int64_t) 0xff01ff01ff01ff01,
            (int64_t) 0xff01ff01ff01ff01};
    static const __m512i mask = {
            0x0504070601000302, 0x0d0c0f0e09080b0a,
            0x0504070601000302, 0x0d0c0f0e09080b0a,
            0x0504070601000302, 0x0d0c0f0e09080b0a,
            0x0504070601000302, 0x0d0c0f0e09080b0a};

    u = conditional_negate_epi8(u, Z);
    return _mm512_add_epi8(u, _mm512_shuffle_epi8(u, mask));
}

static inline void complexMultiply(__m512i u, __m512i *ulo, __m512i *uhi) {

    const __m512i indexHiSymmetry = _mm512_set_epi8(
            63, 62, 62, 63, 59, 58, 58,
            59, 55, 54, 54, 55, 51, 50,
            50, 51, 47, 46, 46, 47, 43,
            42, 42, 43, 39, 38, 38, 39,
            35, 34, 34, 35, 31, 30, 30,
            31, 27, 26, 26, 27, 23, 22,
            22, 23, 19, 18, 18, 19, 15,
            14, 14, 15, 11, 10, 10, 11,
            7, 6, 6, 7, 3, 2, 2, 3
    );

    const __m512i indexLoDuplicateReverse = _mm512_set_epi8(
            61, 60, 61, 60, 47, 46, 47,
            46, 53, 52, 53, 52, 49, 48,
            49, 48, 45, 44, 45, 44, 31,
            30, 31, 30, 37, 36, 37, 36,
            33, 32, 33, 32, 29, 28, 29,
            28, 25, 24, 25, 24, 21, 20,
            21, 20, 17, 16, 17, 16, 13,
            12, 13, 12, 9, 8, 9, 8,
            5, 4, 5, 4, 1, 0, 1, 0
    );

    const __m512i negs = _mm512_set_epi16(
            1, -1, 1, -1, 1, -1, 1, -1,
            1, -1, 1, -1, 1, -1, 1, -1,
            1, -1, 1, -1, 1, -1, 1, -1,
            1, -1, 1, -1, 1, -1, 1, -1);

    __m512i vlo, vhi,
    v = _mm512_shuffle_epi8(u, indexHiSymmetry);
    u = _mm512_shuffle_epi8(u, indexLoDuplicateReverse);
    convert_epi8_epi16(u, uhi, ulo);
    convert_epi8_epi16(v, &vhi, &vlo);
    *ulo = _mm512_mullo_epi16(*ulo, vlo);
    *uhi = _mm512_mullo_epi16(*uhi, vhi);

    vlo = _mm512_shufflelo_epi16(_mm512_shufflehi_epi16(*ulo, SHUF_INDEX), SHUF_INDEX); // 2031_4
    *ulo = _mm512_add_epi16(*ulo, conditional_negate_epi16(vlo, negs));
    vhi = _mm512_shufflelo_epi16(_mm512_shufflehi_epi16(*uhi, SHUF_INDEX), SHUF_INDEX); // 2031_4
    *uhi = _mm512_add_epi16(*uhi, conditional_negate_epi16(vhi, negs));
}

static inline __m512 fmDemod(__m512 u) {

//    //_mm512_setr_epi32(4,12,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
//    static const __m512i index = {0x40000000c};
    //_mm512_setr_epi32(5,13,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
    static const __m512i index = {0xd00000005};

    static const __m512 all64s = {
            64.f, 64.f, 64.f, 64.f, 64.f, 64.f, 64.f, 64.f,
            64.f, 64.f, 64.f, 64.f, 64.f, 64.f, 64.f, 64.f};
    static const __m512 all23s = {
            23.f, 23.f, 23.f, 23.f, 23.f, 23.f, 23.f, 23.f,
            23.f, 23.f, 23.f, 23.f, 23.f, 23.f, 23.f, 23.f};
    static const __m512 all41s = {
            41.f, 41.f, 41.f, 41.f, 41.f, 41.f, 41.f, 41.f,
            41.f, 41.f, 41.f, 41.f, 41.f, 41.f, 41.f, 41.f};

    __m512 w, result;

    // Norm
    w = _mm512_mul_ps(u, u);
    w = _mm512_rsqrt14_ps(_mm512_add_ps(w, _mm512_permute_ps(w, 0x1B)));
    u = _mm512_mul_ps(u, w);

    // fast atan2 -> atan2(x,y) = 64y/(23x+41)
    w = _mm512_mul_ps(u, all64s);                  // 64*zj
    u = _mm512_fmadd_ps(all23s, u, all41s);     // 23*zr + 41s
    u = _mm512_mul_ps(w, _mm512_rcp14_ps(_mm512_permute_ps(u, 0x1B)));
    iterations++;
    // NAN check
    result = _mm512_permutexvar_ps(index, _mm512_maskz_and_ps(_mm512_cmp_ps_mask(u, u, 0), u, u));
    return result;
//    result =  _mm512_maskz_and_ps(_mm512_cmp_ps_mask(u, u, 0), u, u);
}

static inline __m512 demodEpi8(__m512i u) {

    static const __m512i negateBIm = {
            (int64_t) 0xff010101ff010101,
            (int64_t) 0xff010101ff010101,
            (int64_t) 0xff010101ff010101,
            (int64_t) 0xff010101ff010101,
            (int64_t) 0xff010101ff010101,
            (int64_t) 0xff010101ff010101,
            (int64_t) 0xff010101ff010101,
            (int64_t) 0xff010101ff010101};

    static const __m512i indexHi = {
            0x2322252423222120,
            0x2726292827262524,
            0x2b2a2d2c2b2a2928,
            0x2f2e31302f2e2d2c,
            0x3332353433323130,
            0x3736393837363534,
            0x3b3a3d3c3b3a3938,
            0x3f3effff3f3e3d3c};

    static const __m512i indexLo = {
            0x302050403020100,
            0x706090807060504,
            0xb0a0d0c0b0a0908,
            0xf0e11100f0e0d0c,
            0x1312151413121110,
            0x1716191815161514,
            0x1b1a1d1c1b1a1918,
            0x1f1e21201f1e1d1c};

    static m512i_pun_t prev;
    __m512 result[8];
    __m512i hi, uhi, ulo;
    m512i_pun_t lo;

    __m512 U[2];

    u = boxcarEpi8(shiftOrigin(u));
    hi = conditional_negate_epi8(_mm512_permutexvar_epi8(indexHi, u), negateBIm);
    lo.v = conditional_negate_epi8(_mm512_permutexvar_epi8(indexLo, u), negateBIm);

    prev.buf[60] = lo.buf[0];
    prev.buf[61] = lo.buf[1];

    complexMultiply(prev.v, &ulo, &uhi);
    convert_epi16_ps(ulo, U);
    result[0] =fmDemod(U[0]);
    result[1] = fmDemod(U[1]);
    convert_epi16_ps(uhi, U);
    result[2] =fmDemod(U[0]);
    result[3] =fmDemod(U[1]);

    complexMultiply(lo.v, &ulo, &uhi);
    convert_epi16_ps(ulo, U);
    result[4] =fmDemod(U[0]);
    result[5] =fmDemod(U[1]);
    convert_epi16_ps(uhi, U);
    result[6] =fmDemod(U[0]);
    result[7] = fmDemod(U[1]);

//    complexMultiply(hi, &ulo, &uhi);
//    convert_epi16_ps(ulo, U);
//    result[6] =fmDemod(U[0]);
//    result[7] =fmDemod(U[1]);

    prev.v = hi;
    return result[3];
}

void *processMatrix(void *ctx) {

    consumerArgs *args = ctx;
    size_t i, j;//, halfLen = DEFAULT_BUF_SIZE >> 7;
    void *buf = _mm_malloc(DEFAULT_BUF_SIZE, 64);
    float *result = _mm_malloc(sizeof(float)*(DEFAULT_BUF_SIZE >> 5), 64); // TODO change this to a float array
    __m512 temp, gain = _mm512_broadcastss_ps(_mm_broadcast_ss(&args->gain));

    while (!args->exitFlag) {
        sem_wait(&args->full);
        pthread_mutex_lock(&args->mutex);
        memcpy(buf, args->buf, DEFAULT_BUF_SIZE);
        pthread_mutex_unlock(&args->mutex);
        sem_post(&args->empty);

        for (i = 0, j = 0; i < DEFAULT_BUF_SIZE; i += 64, ++j) {
            temp = demodEpi8(*(__m512i *) (buf + i));
            result[j] = temp[0];
            result[j + 1] = temp[1];

        }
        if (*(float *) &args->gain) {
            _mm512_mul_ps(*(__m512*)result, gain);
        }
        fwrite(result, sizeof(float), DEFAULT_BUF_SIZE >> 5, args->outFile);
    }

    _mm_free(result);
    _mm_free(buf);
    return NULL;
}

void allocateBuffer(void **buf, const size_t len) {
    *buf = _mm_malloc(len, 64);
}
