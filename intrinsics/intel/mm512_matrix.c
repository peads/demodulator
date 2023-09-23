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
typedef void (*matrixOp512_t)(__m512i, __m64*);

typedef union {
    __m512i v;
    int8_t buf[64];
    int16_t buf16[32];
} m512i_pun_t;

// taken from https://stackoverflow.com/a/55745816
//static inline __m512i conditional_negate_epi16(__m512i target, __m512i signs) {
static inline __m512i conditional_negate_epi16(__m512i target, __m512i signs) {

    static const __m512i ZEROS = {};
    // vpsubw target{k1}, 0, target
    return _mm512_mask_sub_epi16(target, _mm512_movepi16_mask(signs), ZEROS, target);
}

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

    __m512i q1;

    convertInt16ToInt32(&u, &q1);
    ret[0] = _mm512_cvtepi32_ps(u);
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

static inline __m512i boxcarInt16(__m512i u) {

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

static void demod(__m512 *M, __m64 *result) {

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

static inline void demodEpi16(__m512i u, __m64 *result) {

    static const __m512i negateBIm = {
        (int64_t) 0xffff000100010001,
        (int64_t) 0xffff000100010001,
        (int64_t) 0xffff000100010001,
        (int64_t) 0xffff000100010001,
        (int64_t) 0xffff000100010001,
        (int64_t) 0xffff000100010001,
        (int64_t) 0xffff000100010001,
        (int64_t) 0xffff000100010001};

    const __m512i indexHi = _mm512_set_epi16(
        0x1f,0x1e,0xFF,0xFF,0x1f,0x1e,0x1d,0x1c,
        0x1b,0x1a,0x1d,0x1c,0x1b,0x1a,0x19,0x18,
        0x17,0x16,0x19,0x18,0x17,0x16,0x15,0x14,
        0x13,0x12,0x15,0x14,0x13,0x12,0x11,0x10
//        0x33,0x32,0x35,0x34,0x33,0x32,0x31,0x30,
//        0x37,0x36,0x39,0x38,0x37,0x36,0x35,0x34,
//        0x3b,0x3a,0x3d,0x3c,0x3b,0x3a,0x39,0x38,
//        0x3f,0x3e,0xff,0xff,0x3f,0x3e,0x3d,0x3c
        );
    const __m512i indexLo = _mm512_set_epi16(
        0x0f,0x0e,0x11,0x10,0x0f,0x0e,0x0d,0x0c,
        0x0b,0x0a,0x0d,0x0c,0x0b,0x0a,0x09,0x08,
        0x07,0x06,0x09,0x08,0x07,0x06,0x05,0x04,
        0x03,0x02,0x05,0x04,0x03,0x02,0x01,0x00
//        0x13,0x12,0x15,0x14,0x13,0x12,0x11,0x10,
//        0x17,0x16,0x19,0x18,0x15,0x16,0x15,0x14,
//        0x1b,0x1a,0x1d,0x1c,0x1b,0x1a,0x19,0x18,
//        0x1f,0x1e,0x21,0x20,0x1f,0x1e,0x1d,0x1c
        );

    static m512i_pun_t prev;

    __m512i hi;
    m512i_pun_t lo;

    __m512 M[6];
    __m512 temp[2];

    u = boxcarInt16(u);
    hi = conditional_negate_epi16(_mm512_permutexvar_epi16(indexHi, u), negateBIm);
    lo.v = conditional_negate_epi16(_mm512_permutexvar_epi16(indexLo, u), negateBIm);

    prev.buf16[28] = lo.buf16[0];
    prev.buf16[29] = lo.buf16[1];

    convertInt16ToFloat(prev.v, M);
    temp[0] = M[2];
    temp[1] = M[3];

    demod(M, result);
    M[0] = temp[0];
    M[1] = temp[1];
    demod(M, &(result[2]));

    convertInt16ToFloat(lo.v, M);
    temp[0] = M[2];
    temp[1] = M[3];

    demod(M, &(result[4]));
    M[0] = temp[0];
    M[1] = temp[1];
    demod(M, &(result[6]));

    prev.v = hi;
}

static inline void demodEpi8(__m512i u, __m64 *result) {

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

    u = boxcarUint8(convertUint8ToInt8(u));
    hi = conditional_negate_epi8(_mm512_permutexvar_epi8(indexHi, u), negateBIm);
    lo.v = conditional_negate_epi8(_mm512_permutexvar_epi8(indexLo, u), negateBIm);

    prev.buf[60] = lo.buf[0];
    prev.buf[61] = lo.buf[1];

    convertInt8ToFloat(prev.v, M);
    temp[0] = M[2];
    temp[1] = M[3];

    demod(M, result);
    M[0] = temp[0];
    M[1] = temp[1];
    demod(M, &(result[2]));

    convertInt8ToFloat(lo.v, M);
    temp[0] = M[2];
    temp[1] = M[3];

    demod(M, &(result[4]));
    M[0] = temp[0];
    M[1] = temp[1];
    demod(M, &(result[6]));

    prev.v = hi;
}

int processMatrix(FILE *__restrict__ inFile, uint8_t mode, const float inGain,
                  void *__restrict__ outFile) {

    int exitFlag = 0;
    void *buf = _mm_malloc(MATRIX_WIDTH << 4, 64);
    __m64 result[MATRIX_WIDTH << 1];
    size_t elementsRead;

    const uint8_t isGain = inGain != 1.f && fabsf(inGain) > GAIN_THRESHOLD;
    const __m512 gain = _mm512_broadcastss_ps(_mm_broadcast_ss(&inGain));
    const matrixOp512_t demodulate = mode ? demodEpi8 : demodEpi16;

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

        demodulate(*(__m512i *) buf, result);

        if (isGain) {
            _mm512_mul_ps(*(__m512 *) result, gain);
        }

        fwrite(result, OUTPUT_ELEMENT_BYTES, MATRIX_WIDTH << 1, outFile);
    }

    _mm_free(buf);
    return exitFlag;
}
