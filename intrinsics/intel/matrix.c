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

#if 0
    #define MM256_MADD_EPI16(X, Y) _mm256_dpwssd_epi32(_mm256_setzero_si256(), X, Y)
#else
    #define MM256_MADD_EPI16(X, Y) _mm256_madd_epi16(X, Y)
#endif

static inline __m256i shiftOrigin(__m256i u) {

    const __m256i shift = _mm256_setr_epi8(
            -127, -127, -127, -127, -127, -127, -127, -127,
            -127, -127, -127, -127, -127, -127, -127, -127,
            -127, -127, -127, -127, -127, -127, -127, -127,
            -127, -127, -127, -127, -127, -127, -127, -127);

    return _mm256_add_epi8(u, shift);
}

static inline void convert_epi16_ps(__m256i u, __m256i *uhi, __m256i *ulo) {

    *ulo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(u));
    *uhi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(u, 1));
}

static inline void convert_epi8_ps(__m256i u, __m256 *uhi, __m256 *ulo, __m256 *vhi, __m256 *vlo) {

    __m256i temp[2];
    temp[0] = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(u));
    temp[1] = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(u, 1));
    *ulo = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(temp[0])));
    *uhi = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(temp[0], 1)));
    *vlo = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(temp[1])));
    *vhi = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(temp[1], 1)));
}
#if 1
__m128 lp_out_butterWorth_ps(__m128 u) {

    // Degree 8 coefficients
    static const __m128 A = {0.390181f, 0.390181f, 0.390181f, 0.390181f};
    static const __m128 B = {1.11114f, 1.11114f, 1.11114f, 1.11114f};
    static const __m128 C = {1.66294f, 1.66294f, 1.66294f, 1.66294f};
    static const __m128 D = {1.96157f, 1.96157f, 1.96157f, 1.96157f};
    static const __m128 ONES = {1.f, 1.f, 1.f, 1.f};
    // technically, one over omega_c because the normalized, lowpass Butterworth
    // transfer function takes s/omega_c to account for the cutoff
    static const __m128 OMEGA_C = {WC, WC, WC, WC};

    __m128 curr[4], squared, v = _mm_mul_ps(OMEGA_C, u);

    squared = _mm_mul_ps(v, v);
    curr[0] = _mm_add_ps(ONES, _mm_add_ps(squared, _mm_mul_ps(A, v)));
    curr[1] = _mm_add_ps(ONES, _mm_add_ps(squared, _mm_mul_ps(B, v)));
    curr[2] = _mm_add_ps(ONES, _mm_add_ps(squared, _mm_mul_ps(C, v)));
    curr[3] = _mm_add_ps(ONES, _mm_add_ps(squared, _mm_mul_ps(D, v)));

    v = _mm_mul_ps(
            _mm_mul_ps(curr[0], curr[1]),
            _mm_mul_ps(curr[2], curr[3]));

    return _mm_mul_ps(u, _mm_rcp_ps(v));
}

void hp_butterWorth_ps256(__m256i u, __m256i *a, __m256i *b) {

    // Degree 16 Butterworth coefficients in matrix form
    static const __m256 A[] =
            {{0.995185f,0.0980171f,0.995185f,0.0980171f,0.995185f,0.0980171f,0.995185f,0.0980171f},
             {0.95694f,0.290285f,0.95694f,0.290285f,0.95694f,0.290285f,0.95694f,0.290285f},
             {0.881921f,0.471397f,0.881921f,0.471397f,0.881921f,0.471397f,0.881921f,0.471397f},
             {0.77301f,0.634393f,0.77301f,0.634393f,0.77301f,0.634393f,0.77301f,0.634393f},
             {0.634393f,0.77301f,0.634393f,0.77301f,0.634393f,0.77301f,0.634393f,0.77301f},
             {0.471397f,0.881921f,0.471397f,0.881921f,0.471397f,0.881921f,0.471397f,0.881921f},
             {0.290285f,0.95694f,0.290285f,0.95694f,0.290285f,0.95694f,0.290285f,0.95694f},
             {0.0980171f,0.995185f,0.0980171f,0.995185f,0.0980171f,0.995185f,0.0980171f,0.995185f},
             {0.0980171f,-0.995185f,0.0980171f,-0.995185f,0.0980171f,-0.995185f,0.0980171f,-0.995185f},
             {0.290285f,-0.95694f,0.290285f,-0.95694f,0.290285f,-0.95694f,0.290285f,-0.95694f},
             {0.471397f,-0.881921f,0.471397f,-0.881921f,0.471397f,-0.881921f,0.471397f,-0.881921f},
             {0.634393f,-0.77301f,0.634393f,-0.77301f,0.634393f,-0.77301f,0.634393f,-0.77301f},
             {0.77301f,-0.634393f,0.77301f,-0.634393f,0.77301f,-0.634393f,0.77301f,-0.634393f},
             {0.881921f,-0.471397f,0.881921f,-0.471397f,0.881921f,-0.471397f,0.881921f,-0.471397f},
             {0.95694f,-0.290285f,0.95694f,-0.290285f,0.95694f,-0.290285f,0.95694f,-0.290285f},
             {0.995185f,-0.0980171f,0.995185f,-0.0980171f,0.995185f,-0.0980171f,0.995185f,-0.0980171f}};

    __m256i c,d;
    size_t i;
    __m256 uhi,ulo,vhi,vlo,temp[] = {{1,1,1,1,1,1,1,1},
                                     {1,1,1,1,1,1,1,1},
                                     {1,1,1,1,1,1,1,1},
                                     {1,1,1,1,1,1,1,1}};

    static const __m256i indexComplexConjugate = {
            (int64_t) 0xff01ff01ff01ff01,
            (int64_t) 0xff01ff01ff01ff01,
            (int64_t) 0xff01ff01ff01ff01,
            (int64_t) 0xff01ff01ff01ff01
    };

    u = _mm256_sign_epi8(u, indexComplexConjugate);
    convert_epi8_ps(u, &uhi, &ulo, &vhi, &vlo);
    uhi = _mm256_rcp_ps(uhi);
    ulo = _mm256_rcp_ps(ulo);
    vhi = _mm256_rcp_ps(vhi);
    vlo = _mm256_rcp_ps(vlo);

    for (i = 0; i < 16; ++i) {
        temp[0] = _mm256_mul_ps(temp[0], _mm256_sub_ps(ulo, A[i])); // TODO switch addition of the negation?
        temp[1] = _mm256_mul_ps(temp[1], _mm256_sub_ps(uhi, A[i]));
        temp[2] = _mm256_mul_ps(temp[2], _mm256_sub_ps(vlo, A[i]));
        temp[3] = _mm256_mul_ps(temp[3], _mm256_sub_ps(vhi, A[i]));
    }

    for (i = 0; i < 4; ++i) {
        ulo = _mm256_mul_ps(temp[i], temp[i]);
        ulo = _mm256_permute_ps(_mm256_hadd_ps(ulo, ulo), _MM_SHUFFLE(3,1,2,0));
        ulo = _mm256_rcp_ps(ulo);
        temp[i] = _mm256_mul_ps(temp[i],ulo);
    }

    *a = _mm256_cvtps_epi32(temp[0]);
    *b = _mm256_cvtps_epi32(temp[1]);
    c = _mm256_cvtps_epi32(temp[2]);
    d = _mm256_cvtps_epi32(temp[3]);

    // TODO clean up this mess
    *a = _mm256_setr_m128i(
            _mm256_cvtepi32_epi16(*a),
            _mm_cvtepi32_epi16(
                    _mm256_extracti128_si256(*a, 1)));
    *b = _mm256_setr_m128i(
            _mm256_cvtepi32_epi16(*b),
            _mm_cvtepi32_epi16(
                    _mm256_extracti128_si256(*b, 1)));
    *a = _mm256_inserti128_si256(*a, _mm256_castsi256_si128(*b), 1);

    *b = _mm256_setr_m128i(
            _mm256_cvtepi32_epi16(c),
            _mm_cvtepi32_epi16(
                    _mm256_extracti128_si256(c, 1)));
    c = _mm256_setr_m128i(
            _mm256_cvtepi32_epi16(d),
            _mm_cvtepi32_epi16(
                    _mm256_extracti128_si256(d, 1)));
    *b = _mm256_inserti128_si256(*b, _mm256_castsi256_si128(c), 1);
}

#endif

static void hComplexMultiply(__m256i u, __m256i *uhi, __m256i *ulo) {

    static const __m256i indexComplexConjugate = {
            (int64_t) 0x0001ffff00010001,
            (int64_t) 0x0001ffff00010001,
            (int64_t) 0x0001ffff00010001,
            (int64_t) 0x0001ffff00010001
    };
    __m256i tmp;
    hp_butterWorth_ps256(shiftOrigin(u), ulo, uhi);

    // abab
    // cddc
    tmp = _mm256_shufflelo_epi16(_mm256_shufflehi_epi16(*ulo, _MM_SHUFFLE(1, 0, 1, 0)), _MM_SHUFFLE(1, 0, 1, 0));
    *ulo = _mm256_shufflelo_epi16(_mm256_shufflehi_epi16(*ulo, _MM_SHUFFLE(2, 3, 3, 2)), _MM_SHUFFLE(2, 3, 3, 2));
    // because z uhi* = (ac-(-d)b = ac+bd) + I (a(-d)+bc = -ad+bc)
    *ulo = _mm256_sign_epi16(*ulo, indexComplexConjugate);
    *ulo = MM256_MADD_EPI16(tmp, *ulo);

    tmp = _mm256_shufflelo_epi16(_mm256_shufflehi_epi16(*uhi, _MM_SHUFFLE(1, 0, 1, 0)), _MM_SHUFFLE(1, 0, 1, 0));
    *uhi = _mm256_shufflelo_epi16(_mm256_shufflehi_epi16(*uhi, _MM_SHUFFLE(2, 3, 3, 2)), _MM_SHUFFLE(2, 3, 3, 2));
    *uhi = _mm256_sign_epi16(*uhi, indexComplexConjugate);
    *uhi = MM256_MADD_EPI16(tmp, *uhi);
}

static __m256 decimate(__m256i u) {

    __m256 result;
    __m256i uhi, ulo, vhi, vlo;
    __m256i v;
    hComplexMultiply(u, &u, &v);

    // Also serves as the simplest (crudest) lowpass filter (i.e. s[n-1] + s[n])
    // I just like the averaging bit, and it only costs a shift
    v = _mm256_shufflehi_epi16(_mm256_shufflelo_epi16(u, CDAB_INDEX), CDAB_INDEX);
    convert_epi16_ps(v, &vhi, &vlo);
    convert_epi16_ps(u, &uhi, &ulo);
    ulo = _mm256_srai_epi32(_mm256_add_epi32(ulo, vlo), 1);
    uhi = _mm256_srai_epi32(_mm256_add_epi32(uhi, vhi), 1);

    result = _mm256_permutevar8x32_ps(_mm256_blend_ps(_mm256_cvtepi32_ps(ulo),
                    _mm256_cvtepi32_ps(uhi), 0b11001100),
            _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7));
    return result;
}

static __m128 fmDemod(__m256 u) {

    static const __m256 all64s = {64.f, 64.f, 64.f, 64.f, 64.f, 64.f, 64.f, 64.f};
    static const __m256 all23s = {23.f, 23.f, 23.f, 23.f, 23.f, 23.f, 23.f, 23.f};
    static const __m256 all41s = {41.f, 41.f, 41.f, 41.f, 41.f, 41.f, 41.f, 41.f};

    // fast atan2(y,x) := 64y/(23x+41*Sqrt[x^2+y^2])
    __m256 v = _mm256_mul_ps(u, u),
            hypot = _mm256_permute_ps(v, _MM_SHUFFLE(2, 3, 0, 1));
    hypot = _mm256_add_ps(v, hypot);
    hypot = _mm256_sqrt_ps(hypot);

    // 64*y
    v = _mm256_fmadd_ps(all23s, u, _mm256_mul_ps(all41s, hypot));
    // 1/(23*x+41*hypot)
    v = _mm256_permute_ps(_mm256_rcp_ps(v), _MM_SHUFFLE(2, 3, 0, 1));
    // 64*y/(23*x*41*hypot)
    u = _mm256_mul_ps(_mm256_mul_ps(all64s, u), v);

    // NAN check
    u = _mm256_and_ps(u, _mm256_cmp_ps(u, u, 0));

    return _mm256_castps256_ps128(_mm256_permutevar8x32_ps(u,
            _mm256_setr_epi32(1, 3, 5, 7, 0, 2, 3, 6)));
}

void *processMatrix(void *ctx) {

    consumerArgs *args = ctx;
    size_t i;
    __m128 result = {};
    uint8_t *buf = _mm_malloc(DEFAULT_BUF_SIZE, ALIGNMENT);

    while (!args->exitFlag) {
        sem_wait(&args->full);
        pthread_mutex_lock(&args->mutex);
        memcpy(buf, args->buf, DEFAULT_BUF_SIZE);
        pthread_mutex_unlock(&args->mutex);
        sem_post(&args->empty);

        for (i = 0; i < DEFAULT_BUF_SIZE; i += 32) {
            result = fmDemod(decimate(*(__m256i *) (buf + i)));
#ifdef USE_BUTTERWORTH
//            result = butterWorth_ps(result);
#endif
            fwrite(&result, sizeof(__m128), 1, args->outFile);
        }
    }

    _mm_free(buf);
    return NULL;
}

void allocateBuffer(void **buf, const size_t len) {

    *buf = _mm_malloc(len, ALIGNMENT);
}
