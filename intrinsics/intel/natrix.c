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

typedef __m512i (*vectorOp512_t)(__m512i);
typedef __m128 (*vectorOp128_t)(__m128i);

typedef struct {
    vectorOp512_t boxcar;
    vectorOp512_t convertIn;
    vectorOp128_t convertOut;
} vectorOps_t;

// taken from https://stackoverflow.com/a/55745816
static inline __m512i conditional_negate_epi16(__m512i target, __m512i signs) {
    // vpsubw target{k1}, 0, target
    return _mm512_mask_sub_epi16(target, _mm512_movepi16_mask(signs), _mm512_setzero_si512(), target);
}

static inline __m512i conditional_negate_epi8(__m512i target, __m512i signs) {
    // vpsubw target{k1}, 0, target
    return _mm512_mask_sub_epi8(target, _mm512_movepi8_mask(signs), _mm512_setzero_si512(), target);
}

static inline __m128 convertInt16ToFloat(__m128i u) {

    return _mm_cvtepi32_ps(_mm_cvtepi16_epi32(u));
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

static inline __m128 convertInt8ToFloat(__m128i u) {

    return convertInt16ToFloat(_mm_cvtepi8_epi16(u));
}

static inline __m512i boxcarUint8(__m512i u) {

    static const __m256i Z = {
        (int64_t) 0xff01ff01ff01ff01,
        (int64_t) 0xff01ff01ff01ff01,
        (int64_t) 0xff01ff01ff01ff01,
        (int64_t) 0xff01ff01ff01ff01};
    static const __m512i mask = {
        0x0504070601000302, 0x0d0c0f0e09080b0a,
        0x0504070601000302, 0x0d0c0f0e09080b0a,
        0x0504070601000302, 0x0d0c0f0e09080b0a,
        0x0504070601000302, 0x0d0c0f0e09080b0a};

    u = conditional_negate_epi8(u, _mm512_inserti64x4(_mm512_castsi256_si512(Z), Z, 1));
    return _mm512_add_epi8(u,  _mm512_shuffle_epi8(u, mask));
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

static inline __m256 gather(__m128 u, __m128 v) {

    static const __m256 negateBIm = {1.f, 1.f, 1.f, -1.f, 1.f, 1.f, 1.f, -1.f}; //0x010101FF...

    return _mm256_mul_ps(_mm256_set_m128(_mm_blend_ps(u, v, 0b0011), u), negateBIm);
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

static float fmDemod(__m128 x) {

    static const __m256 all64s = {64.f, 64.f, 64.f, 64.f, 64.f, 64.f, 64.f, 64.f};
    static const __m256 all23s = {23.f, 23.f, 23.f, 23.f, 23.f, 23.f, 23.f, 23.f};
    static const __m256 all41s = {41.f, 41.f, 41.f, 41.f, 41.f, 41.f, 41.f, 41.f};

    static __m128 prev = {0.f, 0.f, 0.f, 0.f};

    __m128 u0 = prev;
    __m128 v0 = prev = x;
    __m256 v, w, y;
    __m256 u = gather(u0, v0); // {ar, aj, br, bj, cr, cj, br, bj}

    preNormMult(&u, &v);
    preNormAddSubAdd(&u, &v, &w);

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

static inline __m512i nonconversion(__m512i u) {

    return u;
}

static inline int processMode(const uint8_t mode, vectorOps_t *funs) {

    switch (mode) {
        case 0: // default mode (input int16)
            funs->boxcar = boxcarInt16;
            funs->convertIn = nonconversion;
            funs->convertOut = convertInt16ToFloat;
            break;
        case 1: // input uint8
            funs->boxcar = boxcarUint8;
            funs->convertIn = convertUint8ToInt8;
            funs->convertOut = convertInt8ToFloat;
            break;
        default:
            return -1;
    }
    return 0;
}

int processMatrix(FILE *__restrict__ inFile, uint8_t mode, const float inGain,
                  void *__restrict__ outFile) {

    vectorOps_t *funs = malloc(sizeof(*funs));
    int exitFlag = processMode(mode, funs);
    size_t elementsRead;
    void *buf = _mm_malloc(MATRIX_WIDTH << 4, 64);
    float result[MATRIX_WIDTH << 1] __attribute__((aligned(64)));
    __m128i lolo, lohi, hilo, hihi;
    __m512i v;

    const uint8_t isGain = fabsf(1.f - inGain) > GAIN_THRESHOLD;
    const __m256 gain = _mm256_broadcast_ss(&inGain);

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

        v = funs->boxcar(funs->convertIn((*(__m512i *) buf)));

        lolo = _mm512_castsi512_si128(v);
        lohi = _mm512_extracti64x2_epi64(v, 1);
        hilo = _mm512_extracti64x2_epi64(v, 2);
        hihi = _mm512_extracti64x2_epi64(v, 3);

        result[0] = fmDemod(funs->convertOut(_mm_unpacklo_epi64(lolo, lolo)));
        result[1] = fmDemod(funs->convertOut(_mm_unpackhi_epi64(lolo, lolo)));
        result[2] = fmDemod(funs->convertOut(_mm_unpacklo_epi64(lohi, lohi)));
        result[3] = fmDemod(funs->convertOut(_mm_unpackhi_epi64(lohi, lohi)));
        result[4] = fmDemod(funs->convertOut(_mm_unpacklo_epi64(hilo, hilo)));
        result[5] = fmDemod(funs->convertOut(_mm_unpackhi_epi64(hilo, hilo)));
        result[6] = fmDemod(funs->convertOut(_mm_unpacklo_epi64(hihi, hihi)));
        result[7] = fmDemod(funs->convertOut(_mm_unpackhi_epi64(hihi, hihi)));

        if (isGain) {
            _mm256_mul_ps(*(__m256 *) &result, gain);
        }

        fwrite(result, OUTPUT_ELEMENT_BYTES, MATRIX_WIDTH << 1, outFile);
    }

    _mm_free(buf);
    free(funs);

    return exitFlag;
}
