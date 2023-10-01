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
#include <pthread.h>
#include <semaphore.h>
#include <string.h>
#include "definitions.h"
#include "matrix.h"

typedef void (*matrixOp512_t)(__m512i, __m64 *__restrict__);

typedef union {
    __m512i v;
    int8_t buf[64];
    int16_t buf16[32];
} m512i_pun_t;

static uint8_t gMode;
static FILE *gOutFile;
static int exitFlag;
static __m512 gGain;
static __m64 *gResult;
static matrixOp512_t demodFun;
static size_t elementsRead;
static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
sem_t full, empty;

// taken from https://stackoverflow.com/a/55745816
static inline __m512i conditional_negate_epi16(__m512i target, __m512i signs) {

    static const __m512i ZEROS = {};
    // vpsubw target{k1}, 0, target
    return _mm512_mask_sub_epi16(target, _mm512_movepi16_mask(signs), ZEROS, target);
}

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

static inline void convert_epi8_epi16(__m512i *__restrict__ u, __m512i *__restrict__ v) {

    *v = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(*u, 1));
    *u = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(*u));
}

static inline void convert_epi16_epi32(__m512i *__restrict__ u, __m512i *__restrict__ v) {

    *v = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(*u, 1));
    *u = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(*u));
}

static inline void convert_epi16_ps(__m512i u, __m512 *__restrict__ ret) {

    __m512i q1;

    convert_epi16_epi32(&u, &q1);
    ret[0] = _mm512_cvtepi32_ps(u);
    ret[1] = _mm512_cvtepi32_ps(q1);
}

static inline void convert_epi8_ps(__m512i u, __m512 *__restrict__ ret) {

    __m512i v = {};
    convert_epi8_epi16(&u, &v);
    convert_epi16_ps(u, ret);
    convert_epi16_ps(v, &(ret[2]));
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

static inline __m512i boxcarEpi16(__m512i u) {

    static const __m512i Z = {
            (int64_t) 0xffff0001ffff0001,
            (int64_t) 0xffff0001ffff0001,
            (int64_t) 0xffff0001ffff0001,
            (int64_t) 0xffff0001ffff0001,
            (int64_t) 0xffff0001ffff0001,
            (int64_t) 0xffff0001ffff0001,
            (int64_t) 0xffff0001ffff0001,
            (int64_t) 0xffff0001ffff0001};
    static const __m512i mask = {
            0x0302010007060504, 0x0b0a09080f0e0d0c,
            0x0302010007060504, 0x0b0a09080f0e0d0c,
            0x0302010007060504, 0x0b0a09080f0e0d0c,
            0x0302010007060504, 0x0b0a09080f0e0d0c};
    u = conditional_negate_epi16(u, Z);
    return _mm512_add_epi16(u, _mm512_shuffle_epi8(u, mask));
}

static inline void preNormMult(__m512 *__restrict__ u, __m512 *__restrict__ v) {

    //  {bj, br, br, bj, bj, br, br, bj} *
    //  {aj, aj, ar, ar, cj, cj, cr, cr}
    // = {aj*bj, aj*br, ar*br, ar*bj, bj*cj, br*cj, br*cr, bj*cr}
    *v = _mm512_permute_ps(*u, 0xEB);
    *u = _mm512_mul_ps(_mm512_permute_ps(*u, 0x5), *v);
}

static inline void
preNormAddSubAdd(__m512 *__restrict__ u, __m512 *__restrict__ v, __m512 *__restrict__ w) {

    static const __m512 ONES = {
            1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f,
            1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f};

    // {aj, bj, ar, br, cj, dj, cr, dr}
    // {ar-aj, aj+bj, br-ar, bj+br, cr-cj, cj+dj, dr-cr, dj+dr}
    // {(ar-aj)^2, (aj+bj)^2, (br-ar)^2, (bj+br)^2, (cr-cj)^2, (cj+dj)^2, (dr-cr)^2, (dj+dr)^2}
    // {ar^2, aj^2, br^2, bj^2, cr^2, cj^2, dr^2, dj^2} + {bj^2, br^2, aj^2, ar^2, ... }
    // = {ar^2+bj^2, aj^2+br^2, br^2+aj^2, bj^2+ar^2, ... }
    *w = _mm512_permute_ps(*u, 0x8D);
    *u = _mm512_fmaddsub_ps(ONES, *u, *w);
    *v = _mm512_mul_ps(*u, *u);
    *w = _mm512_permute_ps(*v, 0x1B);
    *v = _mm512_add_ps(*v, *w);
}

static __m512 fmDemod(__m512 u, __m512 v, __m512 w) {

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

    __m512 y;

    // Norm
    v = _mm512_sqrt_ps(v);
    u = _mm512_mul_ps(u, v);

    // fast atan2 -> atan2(x,y) = 64y/(23x+41)
    w = _mm512_mul_ps(u, all64s);                  // 64*zj
    u = _mm512_fmadd_ps(all23s, u, all41s);     // 23*zr + 41s
    y = _mm512_rcp14_ps(_mm512_permute_ps(u, 0x1B));
    u = _mm512_mul_ps(w, y);

    // NAN check
    return _mm512_permutexvar_ps(index, _mm512_maskz_and_ps(_mm512_cmp_ps_mask(u, u, 0), u, u));
}

static inline void demod(__m512 *__restrict__ M, __m64 *__restrict__ result) {

    __m512 res;

    preNormMult(M, &(M[2]));
    preNormMult(&(M[1]), &(M[3]));

    preNormAddSubAdd(&M[0], &M[2], &M[4]);
    preNormAddSubAdd(&M[1], &M[3], &M[5]);

    res = fmDemod(M[0], M[2], M[4]);
    result[0] = *(__m64 *) &res;
    res = fmDemod(M[1], M[3], M[5]);
    result[1] = *(__m64 *) &res;
}

static inline void demodEpi16(__m512i u, __m64 *__restrict__ result) {

    static const __m512i negateBIm = {
            (int64_t) 0xffff000100010001,
            (int64_t) 0xffff000100010001,
            (int64_t) 0xffff000100010001,
            (int64_t) 0xffff000100010001,
            (int64_t) 0xffff000100010001,
            (int64_t) 0xffff000100010001,
            (int64_t) 0xffff000100010001,
            (int64_t) 0xffff000100010001};

    static const __m512i indexHi = {
            0x13001200110010,
            0x13001200150014,
            0x17001600150014,
            0x17001600190018,
            0x1b001a00190018,
            0x1b001a001d001c,
            0x1f001e001d001c,
            0x1f001e00FF00FF};

    static const __m512i indexLo = {
            0x3000200010000,
            0x3000200050004,
            0x7000600050004,
            0x7000600090008,
            0xb000a00090008,
            0xb000a000d000c,
            0xf000e000d000c,
            0xf000e00110010};

    static m512i_pun_t prev;

    __m512i hi;
    m512i_pun_t lo;

    __m512 M[6];

    u = boxcarEpi16(u);
    hi = conditional_negate_epi16(_mm512_permutexvar_epi16(indexHi, u), negateBIm);
    lo.v = conditional_negate_epi16(_mm512_permutexvar_epi16(indexLo, u), negateBIm);

    prev.buf16[28] = lo.buf16[0];
    prev.buf16[29] = lo.buf16[1];

    convert_epi16_ps(prev.v, M);
    demod(M, result);

    convert_epi16_ps(lo.v, M);
    demod(M, &(result[2]));

    convert_epi16_ps(hi, M);
    demod(M, &(result[2]));

    prev.v = hi;
}

static inline void demodEpi8(__m512i u, __m64 *__restrict__ result) {

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

static void demodulate(void *buf,
                       const matrixOp512_t fun,
                       __m64 *result,
                       const size_t iterations,
                       __m512 gain,
                       FILE *outFile,
                       const uint8_t mode) {

    size_t i;
    size_t shiftIndex;
    for (i = 0; i < iterations; ++i) {
        shiftIndex = i << 2;
        fun(*(__m512i *) (buf + (shiftIndex << 3) * iterations), &(result[shiftIndex]));

        if (*(float *) &gain) {
            _mm512_mul_ps(*(__m512 *) result, gain);
        }

        fwrite(result, OUTPUT_ELEMENT_BYTES, MATRIX_WIDTH << mode, outFile);
    }
}

void *runProcessMatrix(void *inBuf) {

    void *buf = _mm_malloc(DEFAULT_BUF_SIZE >> gMode, 64);

    while (!exitFlag) {
        sem_wait(&full);
        pthread_mutex_lock(&mutex);
        memcpy(buf, inBuf, DEFAULT_BUF_SIZE >> gMode);
        elementsRead = 0;
        pthread_mutex_unlock(&mutex);
        sem_post(&empty);

        for (int i = 0; i < (DEFAULT_BUF_SIZE >> gMode); i += (128 >> gMode)) {
            demodulate(buf + i, demodFun, gResult, 2 - gMode, gGain, gOutFile, gMode);
        }
    }

    _mm_free(inBuf);
    _mm_free(buf);
    return NULL;
}

int processMatrix(FILE *__restrict__ inFile,
                  const uint8_t mode,
                  float gain,
                  void *__restrict__ outFile) {

    gMode = mode;
    gOutFile = outFile;
    exitFlag = gMode && gMode != 1;
    gResult = _mm_malloc(MATRIX_WIDTH << 1, 64);
    gain = gain != 1.f ? gain : 0.f;
    gGain = _mm512_broadcastss_ps(_mm_broadcast_ss(&gain));
    demodFun = gMode ? demodEpi8 : demodEpi16;
    sem_init(&empty, 0, 1);
    sem_init(&full, 0, 0);

    void *buf = _mm_malloc(DEFAULT_BUF_SIZE >> gMode, 64);
    pthread_t pid;
    elementsRead = 0;

    if (pthread_create(&pid, NULL, runProcessMatrix, buf) != 0) {
        fprintf(stderr, "Unable to create consumer thread\n");
        exit(2);
    }

    while (!exitFlag) {

        sem_wait(&empty);
        pthread_mutex_lock(&mutex);
        elementsRead = fread(buf, 2 - gMode, DEFAULT_BUF_SIZE >> gMode, inFile);

        if ((exitFlag = ferror(inFile))) {
            perror(NULL);
            break;
        } else if (feof(inFile)) {
            exitFlag = EOF;
        }
        pthread_mutex_unlock(&mutex);
        sem_post(&full);
    }

    pthread_join(pid, NULL);
    _mm_free(gResult);
    return exitFlag;
}
