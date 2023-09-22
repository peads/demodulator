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

//typedef __m512i (*vectorOp512_t)(__m512i);
//typedef __m256 (*vectorOp256_t)(__m256i);
//typedef void (*matrixOp512_t)(__m512i, void*);

//typedef struct {
//    vectorOp512_t boxcar;
//    vectorOp512_t convertIn;
//    matrixOp512_t convertOut;
//} vectorOps_t;

typedef union {
    __m512i v;
    int8_t buf[64];
    int16_t buf16[32];
} m512i_pun_t;

//TODO
// taken from https://stackoverflow.com/a/55745816
//static inline __m512i conditional_negate_epi16(__m512i target, __m512i signs) {
//    // vpsubw target{k1}, 0, target
//    return _mm512_mask_sub_epi16(target, _mm512_movepi16_mask(signs), ZEROS, target);
//}

static inline __m512i conditional_negate_epi8(__m512i target, __m512i signs) {
    static const __m512i ZEROS = {};
    // vpsubw target{k1}, 0, target
    return _mm512_mask_sub_epi8(target, _mm512_movepi8_mask(signs), ZEROS, target);
}

//TODO
//static inline __m512* convertInt16ToFloat(__m256i u) {
//
//    return _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(u)));
//}

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

    *v = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(*u,1));
    *u = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(*u));
}

static inline void convertInt16ToFloat(__m512i u, /*__m512i v,*/ __m512 *ret) {

    __m512i q0, q1;//, q2, q3;

    q0 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(u));
    q1 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(u, 1));

//    q2 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v));
//    q3 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(v, 1));

    ret[0] = _mm512_cvtepi32_ps(q0);
    ret[1] = _mm512_cvtepi32_ps(q1);
//    ret[2] = _mm512_cvtepi32_ps(q2);
//    ret[3] = _mm512_cvtepi32_ps(q3);
}

//static inline void convertInt8ToFloat(__m512i u, __m512 *ret) {
//
//    static __m512 prev = {
//        0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,
//        0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
//    __m512i v = {};
//    convertInt8ToInt16(&u, &v);
//    convertInt16ToFloat(u, v, ret);
//}

//TODO
//static inline __m512i nonconversion(__m512i u) {
//
//    return u;
//}

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
    return _mm512_add_epi8(u,  _mm512_shuffle_epi8(u, mask));
}

// TODO
//static inline __m512i boxcarInt16(__m512i u) {
//
//    static const __m512i Z = {
//        (int64_t) 0xffff0001ffff0001,
//        (int64_t) 0xffff0001ffff0001,
//        (int64_t) 0xffff0001ffff0001,
//        (int64_t) 0xffff0001ffff0001,
//        (int64_t) 0xffff0001ffff0001,
//        (int64_t) 0xffff0001ffff0001,
//        (int64_t) 0xffff0001ffff0001,
//        (int64_t) 0xffff0001ffff0001};
//    static const __m512i mask = {
//        0x0302010007060504, 0x0b0a09080f0e0d0c,
//        0x0302010007060504, 0x0b0a09080f0e0d0c,
//        0x0302010007060504, 0x0b0a09080f0e0d0c,
//        0x0302010007060504, 0x0b0a09080f0e0d0c};
//    u = conditional_negate_epi16(u, Z);
//    return _mm512_add_epi16(u, _mm512_shuffle_epi8(u, mask));
//}

//static inline void preNormMult(__m512 *u, __m512 *v) {
//
//    *v = _mm512_permute_ps(*u, 0xEB);   //  {bj, br, br, bj, bj, br, br, bj} *
//                                        //  {aj, aj, ar, ar, cj, cj, cr, cr}
//                                        // = {aj*bj, aj*br, ar*br, ar*bj, bj*cj, br*cj, br*cr, bj*cr}
//    *u = _mm512_permute_ps(*u, 0x5);
//    *u = _mm512_mul_ps(*u, *v);
//}

//static inline void preNormAddSubAdd(__m512 *u, __m512 *v, __m512 *w) {
//
//    static const __m512 ONES = {
//        1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,
//        1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f};
//
//    *w = _mm512_permute_ps(*u, 0x8D); // {aj, bj, ar, br, cj, dj, cr, dr}
//    *u = _mm512_fmaddsub_ps(//TODO consider looking for separate add/sub mask intrinsics
//        ONES, *u, *w);      // {ar-aj, aj+bj, br-ar, bj+br, cr-cj, cj+dj, dr-cr, dj+dr}
//    *v = _mm512_mul_ps(*u,*u); // {(ar-aj)^2, (aj+bj)^2, (br-ar)^2, (bj+br)^2, (cr-cj)^2, (cj+dj)^2, (dr-cr)^2, (dj+dr)^2}
//    *w = _mm512_permute_ps(*v, 0x1B); // {ar^2, aj^2, br^2, bj^2, cr^2, cj^2, dr^2, dj^2} +
//                                      // {bj^2, br^2, aj^2, ar^2, ... }
//    *v = _mm512_add_ps(*v, *w);// = {ar^2+bj^2, aj^2+br^2, br^2+aj^2, bj^2+ar^2, ... }
//}

static __m512 fmDemod(__m512 u, __m512 v,  __m512 w) {

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

    const __m512i indexV = _mm512_set_epi8( // _MM_SHUFFLE(3,2,2,3) = 0xEB
        63,62,62,63,
        59,58,58,59,
        55,54,54,55,
        51,50,50,51,
        47,46,46,47,
        43,42,42,43,
        39,38,38,39,
        35,34,34,35,
        31,30,30,31,
        27,26,26,27,
        23,22,22,23,
        19,18,18,19,
        15,14,14,15,
        11,10,10,11,
        7, 6, 6, 7,
        3, 2, 2, 3);

    const __m512i indexU = _mm512_set_epi8( // _MM_SHUFFLE(0,0,1,1) = 0x5
        62,62,61,61,
        58,58,57,57,
        54,54,53,53,
        50,50,49,49,
        46,46,45,45,
        42,42,41,41,
        38,38,37,37,
        34,34,33,33,
        30,30,29,29,
        26,26,25,25,
        20,20,21,21,
        16,16,17,17,
        12,12,13,13,
        8,8,9,9,
        4,4,5,5,
        0,0,1,1
    );

    const __m512i indexW = _mm512_set_epi16(
        30,28,31,29,
        26,24,27,25,
        22,20,23,21,
        18,16,19,17,
        14,12,15,13,
        10,8,11,9,
        6,4,7,5,
        2,0,3,1
    );

    const __m512i reverseIndex = _mm512_set_epi16(
        0,1,2,3,4,5,6,7,8,9,
        10,11,12,13,14,15,16,17,18,19,
        20,21,22,23,24,25,26,27,28,29,
        30,31);

    __m512 ret[4];
    __m512 retW[2];
    __m512 res;
    __m512i u, v, w, vLo16, vHi16, uLo16, uHi16;

    //  {bj, br, br, bj, bj, br, br, bj} *
    //  {aj, aj, ar, ar, cj, cj, cr, cr}
    // = {aj*bj, aj*br, ar*br, ar*bj, bj*cj, br*cj, br*cr, bj*cr}
    v = _mm512_permutexvar_epi8(indexV, b);
    u = _mm512_permutexvar_epi8(indexU, b);

    uLo16 = u;
    vLo16 = v;
    convertInt8ToInt16(&uLo16, &uHi16);
    convertInt8ToInt16(&vLo16, &vHi16);
    uLo16 = _mm512_mullo_epi16(uLo16, vLo16);
    uHi16 = _mm512_mullo_epi16(uHi16, vHi16);

    // {aj, bj, ar, br, cj, dj, cr, dr}
    //
    // {ar-aj, aj+bj, br-ar, bj+br, cr-cj, cj+dj, dr-cr, dj+dr}
    // {(ar-aj)^2, (aj+bj)^2, (br-ar)^2, (bj+br)^2, (cr-cj)^2, (cj+dj)^2, (dr-cr)^2, (dj+dr)^2}
    //
    // {ar^2, aj^2, br^2, bj^2, cr^2, cj^2, dr^2, dj^2} + {bj^2, br^2, aj^2, ar^2, ... }
    // = {ar^2+bj^2, aj^2+br^2, br^2+aj^2, bj^2+ar^2, ... }
    u = uLo16;
    w = _mm512_permutexvar_epi16(indexW, u);         // _MM_SHUFFLE(2,0,3,1) = 0x8D
    u = _mm512_mask_add_epi16(u, 0xAAAAAAAA, u, w);
    u = _mm512_mask_sub_epi16(u, 0x55555555, u, w);
    v = _mm512_mullo_epi16(u,u);
    w = _mm512_permutexvar_epi16(reverseIndex, v);    // _MM_SHUFFLE(0,1,2,3) = 0x1B
    v = _mm512_add_epi16(v, w);
    v = _mm512_mask_set1_epi16(v, _mm512_cmp_epi16_mask(_mm512_setzero_si512(), v, _MM_CMPINT_NLT), 0);
//    v = _mm512_mask_set1_epi16(v, _mm512_cmp_epi16_mask(_mm512_setzero_si512(), v, _MM_CMPINT_NLT), INT16_MAX);


    convertInt16ToFloat(u,ret);
    convertInt16ToFloat(v,&(ret[2]));
    convertInt16ToFloat(w, retW);

    res = fmDemod(ret[0], ret[2], retW[0]);
    result[0] = *(__m64*)&res;
    res = fmDemod(ret[1], ret[3], retW[1]);
    result[1] = *(__m64*)&res;

    u = uHi16;
    w = _mm512_permutexvar_epi16(indexW, u);         // _MM_SHUFFLE(2,0,3,1) = 0x8D
    u = _mm512_mask_add_epi16(u, 0xAAAAAAAA, u, w);
    u = _mm512_mask_sub_epi16(u, 0x55555555, u, w);
    v = _mm512_mullo_epi16(u, u);
    w = _mm512_permutexvar_epi16(reverseIndex, v);    // _MM_SHUFFLE(0,1,2,3) = 0x1B
    v = _mm512_add_epi16(v, w);
    v = _mm512_mask_set1_epi16(v,_mm512_cmp_epi16_mask(_mm512_setzero_si512(), v, _MM_CMPINT_NLT),0);
//    v = _mm512_mask_set1_epi16(v, _mm512_cmp_epi16_mask(_mm512_setzero_si512(), v, _MM_CMPINT_NLT), INT16_MAX);

    convertInt16ToFloat(u,ret);
    convertInt16ToFloat(v,&(ret[2]));
    convertInt16ToFloat(w, retW);

    res = fmDemod(ret[0], ret[2], retW[0]);
    result[2] = *(__m64*)&res;
    res = fmDemod(ret[1], ret[3], retW[1]);
    result[3] = *(__m64*)&res;
}

static inline void demod(__m512i u, __m64 *result) {

    static const __m512i negateBIm = {
        (int64_t) 0xff010101ff010101,
        (int64_t) 0xff010101ff010101,
        (int64_t) 0xff010101ff010101,
        (int64_t) 0xff010101ff010101,
        (int64_t) 0xff010101ff010101,
        (int64_t) 0xff010101ff010101,
        (int64_t) 0xff010101ff010101,
        (int64_t) 0xff010101ff010101};

    const __m512i indexA = _mm512_set_epi8(
        63,62,/*next_1*/-1,/*next_0*/-1, 63,62,61,60,
        59,58,61,60, 59,58,57,56,
        55,54,57,56, 55,54,53,52,
        51,50,53,52, 51,50,49,48,
        47,46,49,48, 47,46,45,44,
        43,42,45,44, 43,42,41,40,
        39,38,41,40, 39,38,37,36,
        35,34,37,36, 35,34,33,32);

    const __m512i indexB = _mm512_set_epi8(
        31,30,33,32, 31,30,29,28,
        27,26,29,28, 27,26,25,24,
        23,22,25,24, 23,22,21,20,
        19,18,21,20, 19,18,17,16,
        15,14,17,16, 15,14,13,12,
        11,10,13,12, 11,10, 9, 8,
        7, 6, 9, 8,  7, 6, 5, 4,
        3, 2, 5, 4,  3, 2, 1, 0);

    static m512i_pun_t prev;
    __m512i a = conditional_negate_epi8(_mm512_permutexvar_epi8(indexA, u), negateBIm);
    // TODO prev[60]=b[0], prev[61]=b[1]
    m512i_pun_t b = {conditional_negate_epi8(_mm512_permutexvar_epi8(indexB, u), negateBIm)};

    prev.buf[60] = b.buf[0];
    prev.buf[61] = b.buf[1];

    demod2(prev.v, result);
    prev.v = a;
    demod2(b.v, &(result[4]));
}

// TODO
//static inline int processMode(const uint8_t mode, vectorOps_t *funs) {
//
//    switch (mode) {
//        //TODO
////        case 0: // default mode (input int16)
////            funs->boxcar = boxcarInt16;
////            funs->convertIn = nonconversion;
////            funs->convertOut = convertInt16ToFloat;
////            break;
//        case 1: // input uint8
//            funs->boxcar = boxcarUint8;
//            funs->convertIn = convertUint8ToInt8;
//            funs->convertOut = convertInt8ToFloat;
//            break;
//        default:
//            return -1;
//    }
//    return 0;
//}

int processMatrix(FILE *__restrict__ inFile, uint8_t mode, const float inGain,
                  void *__restrict__ outFile) {
//TODO
//    vectorOps_t *funs = malloc(sizeof(*funs));
    int exitFlag = 0;//processMode(mode, funs);
    void *buf = _mm_malloc(MATRIX_WIDTH << 4, 64);
    __m64 *result = _mm_malloc(MATRIX_WIDTH << 1, 64);
//    __m64 result[MATRIX_WIDTH << 1];


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
            _mm512_mul_ps(*(__m512 *)result, gain);
        }

        fwrite(result, OUTPUT_ELEMENT_BYTES, MATRIX_WIDTH << 1, outFile);
    }

    _mm_free(result);
    _mm_free(buf);
//    free(funs); //TODO

    return exitFlag;
}
