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
#include "matrix.h"

// taken from https://stackoverflow.com/a/55745816
static inline __m512i conditional_negate_epi8(__m512i target, __m512i signs) {

    static const __m512i ZEROS = {};
    // vpsubb target{k1}, 0, target
    return _mm512_mask_sub_epi8(target, _mm512_movepi8_mask(signs), ZEROS, target);
}

static inline __m512i convert_epu8_epi8(__m512i u) {

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

static inline void convert_epi8_ps(__m512i u, __m512 *__restrict__ ret) {

    __m512i w,
            v = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(u, 1));
    u = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(u));

    w = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(u, 1));
    u = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u));

    ret[0] = _mm512_cvtepi32_ps(u);
    ret[1] = _mm512_cvtepi32_ps(w);

    w = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(v, 1));
    v = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v));

    ret[2] = _mm512_cvtepi32_ps(v);
    ret[3] = _mm512_cvtepi32_ps(w);
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

static inline __m512 complexMult(__m512 u) {

    static const __m512 ONES = {
            1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f,
            1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
    //  {bj, br, br, bj, bj, br, br, bj} *
    //  {aj, aj, ar, ar, cj, cj, cr, cr}
    // = {aj*bj, aj*br, ar*br, ar*bj, bj*cj, br*cj, br*cr, bj*cr}
    // {aj, bj, ar, br, cj, dj, cr, dr}
    // {ar-aj, aj+bj, br-ar, bj+br, cr-cj, cj+dj, dr-cr, dj+dr}
    __m512 v = _mm512_permute_ps(u, 0xEB);
    u = _mm512_mul_ps(_mm512_permute_ps(u, 0x5), v);
    return _mm512_fmaddsub_ps(ONES, u, _mm512_permute_ps(u, 0x8D));
}

static inline __m512 fmDemod(__m512 u) {

    static const __m512 all64s = {
            64.f, 64.f, 64.f, 64.f, 64.f, 64.f, 64.f, 64.f,
            64.f, 64.f, 64.f, 64.f, 64.f, 64.f, 64.f, 64.f};
    static const __m512 all23s = {
            23.f, 23.f, 23.f, 23.f, 23.f, 23.f, 23.f, 23.f,
            23.f, 23.f, 23.f, 23.f, 23.f, 23.f, 23.f, 23.f};
    static const __m512 all41s = {
            41.f, 41.f, 41.f, 41.f, 41.f, 41.f, 41.f, 41.f,
            41.f, 41.f, 41.f, 41.f, 41.f, 41.f, 41.f, 41.f};

    __m512 w, v;

    u = complexMult(u);
    // Norm
    v = _mm512_mul_ps(u, u);
    v = _mm512_add_ps(v, _mm512_permute_ps(v, 0x1B));
    v = _mm512_sqrt_ps(v);
    u = _mm512_mul_ps(u, v);

    w = _mm512_mul_ps(u, all64s);
    u = _mm512_fmadd_ps(all23s, u, all41s);
    u = _mm512_mul_ps(w, _mm512_rcp14_ps(_mm512_permute_ps(u, 0x1B)));

    // NAN check
    return _mm512_maskz_and_ps(_mm512_cmp_ps_mask(u, u, 0), u, u);
}

static inline void demod(__m512 *__restrict__ M, float *result) {

    __m512 res;

    res = fmDemod(M[0]);
    result[0] = res[5];
    result[1] = res[13];
    res = fmDemod(M[1]);
    result[2] = res[5];
    result[3] = res[13];
}

static inline void demodEpi8(__m512i u, float *result) {

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

    __m512i hi;
    m512i_pun_t lo;

    __m512 M[6];
    __m512 temp[2];

    u = boxcarEpi8(convert_epu8_epi8(u));
    hi = conditional_negate_epi8(_mm512_permutexvar_epi8(indexHi, u), negateBIm);
    lo.v = conditional_negate_epi8(_mm512_permutexvar_epi8(indexLo, u), negateBIm);

    prev.buf[60] = lo.buf[0];
    prev.buf[61] = lo.buf[1];

    convert_epi8_ps(prev.v, M);
    temp[0] = M[2];
    temp[1] = M[3];

    demod(M, result);
    M[0] = temp[0];
    M[1] = temp[1];
    demod(M, &(result[4]));

    convert_epi8_ps(lo.v, M);
    temp[0] = M[2];
    temp[1] = M[3];

    demod(M, &(result[8]));
    M[0] = temp[0];
    M[1] = temp[1];
    demod(M, &(result[12]));

    prev.v = hi;
}

void *processMatrix(void *ctx) {

    consumerArgs *args = ctx;
    size_t i;
    uint8_t *buf = _mm_malloc(DEFAULT_BUF_SIZE, ALIGNMENT);
    float result[DEFAULT_BUF_SIZE >> 3] __attribute__((aligned(ALIGNMENT)));
    __m512 gain = _mm512_broadcastss_ps(_mm_broadcast_ss(&args->gain));

    while (!args->exitFlag) {
        sem_wait(&args->full);
        pthread_mutex_lock(&args->mutex);
        memcpy(buf, args->buf, DEFAULT_BUF_SIZE);
        pthread_mutex_unlock(&args->mutex);
        sem_post(&args->empty);

        for (i = 0; i < DEFAULT_BUF_SIZE; i += 64) {
            demodEpi8(*(__m512i *) (buf + i), result + (i >> 3));

            if (*(float *) &args->gain) {
                _mm512_mul_ps(*(__m512 *) &result, gain);
            }
        }
        fwrite(result, sizeof(float), DEFAULT_BUF_SIZE >> 3, args->outFile);
    }

    _mm_free(buf);
    return NULL;
}

void allocateBuffer(void **buf, const size_t len) {
    *buf = _mm_malloc(len, ALIGNMENT);
}
