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

static __m256 decimate(__m256i u) {

    __m256 result;
    __m256i uhi, ulo, vhi, vlo;
    __m256i v = _mm256_shufflehi_epi16(_mm256_shufflelo_epi16(u, CDAB_INDEX), CDAB_INDEX);
    convert_epi16_ps(v, &vhi, &vlo);
    convert_epi16_ps(u, &uhi, &ulo);
    ulo = _mm256_srai_epi32(_mm256_add_epi32(ulo, vlo), 1);
    uhi = _mm256_srai_epi32(_mm256_add_epi32(uhi, vhi), 1);

    result = _mm256_permutevar8x32_ps(_mm256_blend_ps(_mm256_cvtepi32_ps(ulo),
                    _mm256_cvtepi32_ps(uhi), 0b11001100),
            _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7));
    return result;
}

static __m256i hComplexMultiply(__m256i u) {

    const __m256i indexCD = _mm256_setr_epi8(
            2, 3, 6, 7, 10, 11, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1,
            18, 19, 22, 23, 26, 27, 30, 31, -1, -1, -1, -1, -1, -1, -1, -1
    );
    const __m256i indexAB = _mm256_setr_epi8(
            0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1,
            16, 17, 20, 21, 24, 25, 28, 29, -1, -1, -1, -1, -1, -1, -1, -1
    );
    const __m256i indexDC = _mm256_setr_epi8(
            1, 0, 3, 2, 5, 4, 7, 6,
            9, 8, 11, 10, 13, 12, 15, 14,
            17, 16, 19, 18, 21, 20, 23, 22,
            25, 24, 27, 26, 29, 28, 31, 30
    );
    static const __m256i indexComplexConjugate = {
            (int64_t) 0x01ff01ff01ff01ff,
            (int64_t) 0x01ff01ff01ff01ff,
            (int64_t) 0x01ff01ff01ff01ff,
            (int64_t) 0x01ff01ff01ff01ff
    };
    // ab
    // cd
    __m256i zr, zj,
            v = _mm256_shuffle_epi8(u, indexCD),
            w = _mm256_shuffle_epi8(u, indexAB);

    // ac-(-b)d = ac+bd
    zr = _mm256_dpwssd_epi32(
            _mm256_setzero_si256(),
            _mm256_cvtepi8_epi16(_mm256_castsi256_si128(v)),
            _mm256_cvtepi8_epi16(_mm256_castsi256_si128(w)));
    zr = _mm256_inserti128_si256(zr,
            _mm256_castsi256_si128(
                    _mm256_dpwssd_epi32(_mm256_setzero_si256(),
                            _mm256_cvtepi8_epi16(_mm256_extracti128_si256(v, 1)),
                            _mm256_cvtepi8_epi16(_mm256_extracti128_si256(w, 1)))), 1);

    // ab=>ab
    // cd=>dc
    // a(-d)+bc
    v = _mm256_sign_epi8(_mm256_shuffle_epi8(v, indexDC), indexComplexConjugate);
    zj = _mm256_dpwssd_epi32(
            _mm256_setzero_si256(),
            _mm256_cvtepi8_epi16(_mm256_castsi256_si128(v)),
            _mm256_cvtepi8_epi16(_mm256_castsi256_si128(w)));
    zj = _mm256_inserti128_si256(zj,
            _mm256_castsi256_si128(
                    _mm256_dpwssd_epi32(_mm256_setzero_si256(),
                            _mm256_cvtepi8_epi16(_mm256_extracti128_si256(v, 1)),
                            _mm256_cvtepi8_epi16(_mm256_extracti128_si256(w, 1)))), 1);

    return _mm256_blend_epi16(_mm256_shufflelo_epi16(_mm256_shufflehi_epi16(
            _mm256_alignr_epi8(zj, zj, 0), _MM_SHUFFLE(2, 3, 0, 1)), _MM_SHUFFLE(2, 3, 0, 1)),
            _mm256_alignr_epi8(zr, zr, 0), 0b01010101);
}

#ifndef NO_BUTTERWORTH

static __m128 butterWorth_ps(__m128 u) {

    static const __m128 A = {0.390181f, 0.390181f, 0.390181f, 0.390181f};
    static const __m128 B = {1.11114f, 1.11114f, 1.11114f, 1.11114f};
    static const __m128 C = {1.66294f, 1.66294f, 1.66294f, 1.66294f};
    static const __m128 D = {1.96157f, 1.96157f, 1.96157f, 1.96157f};
    static const __m128 ONES = {1.f, 1.f, 1.f, 1.f};
    // technically, one over omega_c because the normalized Butterworth
    // transfer function takes s/omega_c to account for the cutoff
    static const __m128 OMEGA_C = {0.0008f, 0.0008f, 0.0008f, 0.0008f};

    __m128 curr[4], squared, v = _mm_mul_ps(OMEGA_C, u);

    squared = _mm_mul_ps(v, v);
    curr[0] = _mm_add_ps(ONES, _mm_add_ps(squared, _mm_mul_ps(A, v)));
    curr[1] = _mm_add_ps(ONES, _mm_add_ps(squared, _mm_mul_ps(B, v)));
    curr[2] = _mm_add_ps(ONES, _mm_add_ps(squared, _mm_mul_ps(C, v)));
    curr[3] = _mm_add_ps(ONES, _mm_add_ps(squared, _mm_mul_ps(D, v)));

    v = _mm_mul_ps(
            _mm_mul_ps(curr[0], curr[1]),
            _mm_mul_ps(curr[2], curr[3]));

    return _mm_mul_ps(u, _mm_rcp14_ps(v));
}
#endif

static __m128 fmDemod(__m256 u) {

    static const __m256 all64s = {64.f, 64.f, 64.f, 64.f, 64.f, 64.f, 64.f, 64.f};
    static const __m256 all23s = {23.f, 23.f, 23.f, 23.f, 23.f, 23.f, 23.f, 23.f};
    static const __m256 all41s = {41.f, 41.f, 41.f, 41.f, 41.f, 41.f, 41.f, 41.f};

    // fast atan2 -> atan2(y,x) = 64y/(23x+41*Sqrt[x^2+y^2])
    // 1/23*x+41*hypot
    __m256 v = _mm256_mul_ps(u, u),
            hypot = _mm256_permute_ps(v, _MM_SHUFFLE(2, 3, 0, 1));
    hypot = _mm256_add_ps(v, hypot);
    hypot = _mm256_sqrt_ps(hypot);
    v = _mm256_fmadd_ps(all23s, u, _mm256_mul_ps(all41s, hypot));
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
            result = fmDemod(decimate(hComplexMultiply(shiftOrigin(*(__m256i *) (buf + i)))));
#ifndef NO_BUTTERWORTH
            result = butterWorth_ps(result);
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
