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

#if __AVX512VNNI__
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

static inline void convert_epi8_epi32(__m256i u, __m256 *uhi, __m256 *ulo, __m256 *vhi, __m256 *vlo) {

    __m256i temp[2];
    temp[0] = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(u));
    temp[1] = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(u, 1));
    *ulo = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(temp[0])));
    *uhi = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(temp[0], 1)));
    *vlo = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(temp[1])));
    *vhi = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(temp[1], 1)));
}

#if 1
__m256 lp_out_butterWorth_ps(__m256 u) {

    // Degree 8 coefficients
    static const __m256 A = {
            0.390181f, 0.390181f, 0.390181f, 0.390181f,
            0.390181f, 0.390181f, 0.390181f, 0.390181f};
    static const __m256 B = {
            1.11114f, 1.11114f, 1.11114f, 1.11114f,
            1.11114f, 1.11114f, 1.11114f, 1.11114f};
    static const __m256 C = {
            1.66294f, 1.66294f, 1.66294f, 1.66294f,
            1.66294f, 1.66294f, 1.66294f, 1.66294f};
    static const __m256 D = {
            1.96157f, 1.96157f, 1.96157f, 1.96157f,
            1.96157f, 1.96157f, 1.96157f, 1.96157f};
    static const __m256 ONES = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
    // technically, one over omega_c because the normalized, lowpass Butterworth
    // transfer function takes s/omega_c to scale for the cutoff
    static const __m256 OMEGA_C = {WC, WC, WC, WC, WC, WC, WC, WC};

    __m256 curr[4], squared, v = _mm256_mul_ps(OMEGA_C, u);

    squared = _mm256_mul_ps(v, v);
    curr[0] = _mm256_add_ps(ONES, _mm256_add_ps(squared, _mm256_mul_ps(A, v)));
    curr[1] = _mm256_add_ps(ONES, _mm256_add_ps(squared, _mm256_mul_ps(B, v)));
    curr[2] = _mm256_add_ps(ONES, _mm256_add_ps(squared, _mm256_mul_ps(C, v)));
    curr[3] = _mm256_add_ps(ONES, _mm256_add_ps(squared, _mm256_mul_ps(D, v)));

    v = _mm256_mul_ps(
            _mm256_mul_ps(curr[0], curr[1]),
            _mm256_mul_ps(curr[2], curr[3]));

    return _mm256_mul_ps(u, _mm256_rcp_ps(v));
}

__m256 hp_butterWorth_ps256(__m256i u) {

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

    size_t i;
    __m256  v, a, b = {1, 1, 1, 1, 1, 1, 1, 1};

    static const __m256i indexComplexConjugate = {
            (int64_t) 0xffff000100010001,
            (int64_t) 0xffff000100010001,
            (int64_t) 0xffff000100010001,
            (int64_t) 0xffff000100010001
    };
    v = _mm256_cvtepi32_ps(u);
    a = _mm256_rcp_ps(_mm256_cvtepi32_ps(_mm256_sign_epi16(u, indexComplexConjugate)));

    for (i = 0; i < 16; ++i) {
        b = _mm256_mul_ps(b, _mm256_sub_ps(a, A[i])); // TODO switch addition of the negation?
    }

//  z^-1 = 1/(a+bI) = (a+bI)/|z|^2
    a = _mm256_mul_ps(b, b);
    a = _mm256_permute_ps(_mm256_hadd_ps(a, a), _MM_SHUFFLE(3,1,2,0));
    a = _mm256_rcp_ps(a);
    b = _mm256_mul_ps(b, a);
    v = _mm256_mul_ps(v, b);

    return v;
}

#endif

static __m256i hComplexMultiply(__m256i u) {

    static const __m256i indexComplexConjugate = {
            (int64_t) 0x0001ffff00010001,
            (int64_t) 0x0001ffff00010001,
            (int64_t) 0x0001ffff00010001,
            (int64_t) 0x0001ffff00010001
    };
    __m256i tmp;

    // abab
    // cddc
    tmp = _mm256_shufflelo_epi16(_mm256_shufflehi_epi16(u, _MM_SHUFFLE(1, 0, 1, 0)), _MM_SHUFFLE(1, 0, 1, 0));
    u = _mm256_shufflelo_epi16(_mm256_shufflehi_epi16(u, _MM_SHUFFLE(2, 3, 3, 2)), _MM_SHUFFLE(2, 3, 3, 2));
    // because z w* = (ac-(-d)b = ac+bd) + I (a(-d)+bc = -ad+bc)
    u = _mm256_sign_epi16(u, indexComplexConjugate);
    return MM256_MADD_EPI16(tmp, u);
}

static __m256 fmDemod(__m256 u) {

    static const __m256 all64s = {64.f, 64.f, 64.f, 64.f, 64.f, 64.f, 64.f, 64.f};
    static const __m256 all23s = {23.f, 23.f, 23.f, 23.f, 23.f, 23.f, 23.f, 23.f};
    static const __m256 all41s = {41.f, 41.f, 41.f, 41.f, 41.f, 41.f, 41.f, 41.f};
//    const __m256i index = _mm256_setr_epi32(1, 3, 5, 7, 0, 2, 3, 6);
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
//    u = _mm256_permutevar8x32_ps(u,index);
    return u;
}

__m256i crudeLowpass_epi8_epi16(__m256i u) {

    const __m256i index = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
    u = shiftOrigin(u);
    __m256i v = _mm256_shufflehi_epi16(_mm256_shufflelo_epi16(u, _MM_SHUFFLE(2, 3, 0, 1)),
            _MM_SHUFFLE(2, 3, 0, 1));
    v = _mm256_add_epi8(u, v);
    v = _mm256_shufflehi_epi16(_mm256_shufflelo_epi16(v, _MM_SHUFFLE(2, 0, 3, 1)),
            _MM_SHUFFLE(2, 0, 3, 1));
    v = _mm256_permutevar8x32_epi32(v, index);
    u = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(v));
    return u;
//    u = _mm256_shufflehi_epi16(
//            _mm256_shufflelo_epi16(
//                    _mm256_avg_epu8(u,
//                            _mm256_shufflehi_epi16(
//                                    _mm256_shufflelo_epi16(u, _MM_SHUFFLE(2,3,0,1)),
//                                    _MM_SHUFFLE(2,3,0,1))),
//                    _MM_SHUFFLE(2,0,3,1)),
//            _MM_SHUFFLE(2,0,3,1));
//    u = _mm256_permutevar8x32_epi32(u, index);
//    u = shiftOrigin(u);
//    return _mm256_cvtepi8_epi16(_mm256_castsi256_si128(u));
}

void *processMatrix(void *ctx) {

    consumerArgs *args = ctx;
    size_t i;
    __m256 result = {};
    __m256i s0;
    uint8_t *buf = _mm_malloc(DEFAULT_BUF_SIZE, ALIGNMENT);
    while (!args->exitFlag) {
        sem_wait(&args->full);
        pthread_mutex_lock(&args->mutex);
        memcpy(buf, args->buf, DEFAULT_BUF_SIZE);
        pthread_mutex_unlock(&args->mutex);
        sem_post(&args->empty);

        for (i = 0; i < DEFAULT_BUF_SIZE; i += 32) {
            s0 = crudeLowpass_epi8_epi16(*(__m256i *) (buf + i));
            s0 = hComplexMultiply(s0);
//            result = hp_butterWorth_ps256(s0);
            result = _mm256_cvtepi32_ps(s0);
//            result = _mm256_add_ps(result, _mm256_set1_ps(-2.f));
            result = fmDemod(result);
//            result = lp_out_butterWorth_ps(result);
            fwrite(&result, sizeof(__m256), 1, args->outFile);
        }
    }

    _mm_free(buf);
    return NULL;
}

void allocateBuffer(void **buf, const size_t len) {

    *buf = _mm_malloc(len, ALIGNMENT);
}
