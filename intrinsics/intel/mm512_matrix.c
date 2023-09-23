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
    __m512i v;
    int8_t buf[64];
    int16_t buf16[32];
} m512i_pun_t;

// taken from https://stackoverflow.com/a/55745816
static inline __m512i conditional_negate_epi8(__m512i target, __m512i signs) {

    static const __m512i ZEROS = {};
    // vpsubw target{k1}, 0, target
    return _mm512_mask_sub_epi8(target, _mm512_movepi8_mask(signs), ZEROS, target);
}

static inline __m512i convertUint8ToInt8(__m512i u) {

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

static inline void convertInt8ToInt16(__m512i *u, __m512i *v) {

    *v = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(*u, 1));
    *u = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(*u));
}

static inline void convertInt16ToInt32(__m512i *u, __m512i *v) {

    *v = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(*u, 1));
    *u = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(*u));
}

static inline void convertInt16ToFloat(__m512i u, __m512 *ret) {

    __m512i q0 = u, q1;

    convertInt16ToInt32(&q0, &q1);
    ret[0] = _mm512_cvtepi32_ps(q0);
    ret[1] = _mm512_cvtepi32_ps(q1);
}

static inline void convertInt8ToFloat(__m512i u, __m512 *ret) {

    __m512i v = {};
    convertInt8ToInt16(&u, &v);
    convertInt16ToFloat(u, ret);
    convertInt16ToFloat(v, &(ret[2]));
}

static inline __m512i boxcarUint8(__m512i u) {

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

static inline void preNormMult(__m512 *u, __m512 *v) {

    *v = _mm512_permute_ps(*u, 0xEB);   //  {bj, br, br, bj, bj, br, br, bj} *
                                        //  {aj, aj, ar, ar, cj, cj, cr, cr}
                                        // = {aj*bj, aj*br, ar*br, ar*bj, bj*cj, br*cj, br*cr, bj*cr}
    *u = _mm512_permute_ps(*u, 0x5);
    *u = _mm512_mul_ps(*u, *v);
}

static inline void preNormAddSubAdd(__m512 *u, __m512 *v, __m512 *w) {

    static const __m512 ONES = {
        1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f,
        1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f};

    *w = _mm512_permute_ps(*u, 0x8D);                // {aj, bj, ar, br, cj, dj, cr, dr}
    *u = _mm512_fmaddsub_ps(ONES,*u,*w);   // {ar-aj, aj+bj, br-ar, bj+br, cr-cj, cj+dj, dr-cr, dj+dr}
    *v = _mm512_mul_ps(*u,*u);                // {(ar-aj)^2, (aj+bj)^2, (br-ar)^2, (bj+br)^2, (cr-cj)^2, (cj+dj)^2, (dr-cr)^2, (dj+dr)^2}
    *w = _mm512_permute_ps(*v, 0x1B);               // {ar^2, aj^2, br^2, bj^2, cr^2, cj^2, dr^2, dj^2} +
                                                    // {bj^2, br^2, aj^2, ar^2, ... }
    *v = _mm512_add_ps(*v, *w);               // = {ar^2+bj^2, aj^2+br^2, br^2+aj^2, bj^2+ar^2, ... }
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

static void demod2(__m512i b, __m64 *result) {

    __m512 ret[6];
    __m512 res, u[2];

    convertInt8ToFloat(b, ret);
    u[0] = ret[2];
    u[1] = ret[3];

    preNormMult(ret, &(ret[2]));
    preNormMult(&(ret[1]), &(ret[3]));

    preNormAddSubAdd(&ret[0], &ret[2], &ret[4]);
    preNormAddSubAdd(&ret[1], &ret[3], &ret[5]);

    res = fmDemod(ret[0], ret[2], ret[4]);
    result[0] = *(__m64 *) &res;
    res = fmDemod(ret[1], ret[3], ret[5]);
    result[1] = *(__m64 *) &res;

    preNormMult(u, &(ret[2]));
    preNormMult(&(u[1]), &(ret[3]));

    preNormAddSubAdd(&u[0], &ret[2], &ret[4]);
    preNormAddSubAdd(&u[1], &ret[3], &ret[5]);

    res = fmDemod(u[0], ret[2], ret[4]);
    result[2] = *(__m64 *) &res;
    res = fmDemod(u[1], ret[3], ret[5]);
    result[3] = *(__m64 *) &res;
}

static inline void demod(__m512i u, __m64 *result) {

    static m512i_pun_t prev;

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

    __m512i hi = conditional_negate_epi8(_mm512_permutexvar_epi8(indexHi, u), negateBIm);
    m512i_pun_t lo = {conditional_negate_epi8(_mm512_permutexvar_epi8(indexLo, u), negateBIm)};

    prev.buf[60] = lo.buf[0];
    prev.buf[61] = lo.buf[1];

    demod2(prev.v, result);
    prev.v = hi;
    demod2(lo.v, &(result[4]));
}

int processMatrix(FILE *__restrict__ inFile, uint8_t mode, const float inGain,
                  void *__restrict__ outFile) {

    int exitFlag = 0;//processMode(mode, funs);
    void *buf = _mm_malloc(MATRIX_WIDTH << 4, 64);
    __m64 *result = _mm_malloc(MATRIX_WIDTH << 1, 64);

    size_t elementsRead;
    __m512i v;

    const uint8_t isGain = inGain != 1.f && fabsf(inGain) > GAIN_THRESHOLD;
    const __m512 gain = _mm512_broadcastss_ps(_mm_broadcast_ss(&inGain));

    while (!exitFlag) {

        elementsRead = fread(buf, 1, MATRIX_WIDTH << 4, inFile);

        if ((exitFlag = ferror(inFile))) {
            perror(NULL);
            break;
        } else if (feof(inFile)) {
            exitFlag = EOF;
        } else if (!elementsRead) {
            fprintf(stderr, "This shouldn't happen, but I need to use the result of"
                            "fread. Stupid compiler.");
        }

        v = boxcarUint8(convertUint8ToInt8((*(__m512i *) buf)));
        demod(v, result);

        if (isGain) {
            _mm512_mul_ps(*(__m512 *) result, gain);
        }

        fwrite(result, OUTPUT_ELEMENT_BYTES, MATRIX_WIDTH << 1, outFile);
    }

    _mm_free(result);
    _mm_free(buf);

    return exitFlag;
}
