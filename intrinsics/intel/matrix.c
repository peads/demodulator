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

static inline void convert_epi16_ps(__m256i u, __m256 *uhi, __m256 *ulo) {

    *ulo = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(u)));
    *uhi = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(u, 1)));
}

static __m256 decimate(__m256i u/*TODO ideally this will allow variable decimation iterations*/) {

    __m256 uhi, ulo, vhi, vlo;
    __m256i v = _mm256_shufflehi_epi16(_mm256_shufflelo_epi16(u, CDAB_INDEX), CDAB_INDEX);
    convert_epi16_ps(v, &vhi, &vlo);
    convert_epi16_ps(u, &uhi, &ulo);
    ulo = _mm256_add_ps(ulo, vlo);
    uhi = _mm256_add_ps(uhi, vhi);

    ulo = _mm256_blend_ps(ulo, uhi, 0b11001100);
    ulo = _mm256_permutevar8x32_ps(ulo, _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7));
    return ulo;
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

//__m256i butterWorth_epi8(__m256i u) {
////            {0.0490825, 0.147129, 0.244821, 0.341924, 0.438202,
////                0.533426, 0.627363, 0.71979, 0.810483, 0.899223,
////                0.985796, 1.07, 1.15162, 1.23046, 1.30635, 1.37908,
////                1.44849, 1.51442, 1.57669, 1.63517, 1.68971, 1.74017,
////                1.78645, 1.82842, 1.86599, 1.89906, 1.92755, 1.9514,
////                1.97056, 1.98496, 1.99458, 1.9994}
////    {{
////            0.0981353, 0.293461, 0.48596, 0.67378},
////        {0.85511, 1.02821, 1.1914, 1.34312},
////        {1.4819, 1.60642, 1.71546, 1.80798},
////        {1.88309, 1.94006, 1.97835, 1.99759}}//MatrixForm
//
//    const __m256i Al = _mm256_setr_epi16(
//            0,0,0,0,
//            0,0,0,0,
//            1,1,1,1,
//            1,1,1,1);
//    const __m256i Ar = _mm256_setr_epi16(
//            4,2,1,1,
//            0,0,0,0,
//            0,0,0,0,
//            0,0,0,0);
//    const __m256i ONES = _mm256_setr_epi16(
//            1,1,1,1,
//            1,1,1,1,
//            1,1,1,1,
//            1,1,1,1);
////     const __m256i OMEGA_C = (250.f,250.f,250.f,250.f);
////    static const __m128 OMEGA_C = {0.0008f,0.0008f,0.0008f,0.0008f};
//
//    typedef union {
//        __m256i v;
//        int16_t buf[16];
//    } pun;
//    pun curr[4];
//    __m256i squared;
//
//    int16_t i;
//
//    for (i = 0; i < 4; ++i) {
//        curr[i].v = _mm256_shuffle_epi8(u,
//                        _mm256_setr_epi16(i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i));
//        squared = _mm256_mullo_epi16(curr[i].v,curr[i].v);
//        curr[i].v = _mm256_srav_epi16(curr[i].v, Ar);
//        curr[i].v = _mm256_sllv_epi16(curr[i].v, Al);
//        curr[i].v = _mm256_add_epi16(curr[i].v, squared);
//        curr[i].v = _mm256_add_epi16(ONES, curr[i].v);
////        curr[i] = _mm_mul_ps(OMEGA_C, curr[i]);
//    }
//
////    for (i = 0; i < 4; ++i) {
//    u[0] = curr[0].buf[0];
//    u[1] = curr[1].buf[0];
//    u[2] = curr[2].buf[0];
//    u[3] = curr[3].buf[0];
//    for (i = 1; i < 4; ++i) {
//        u[0] *= curr[0].buf[i];
//        u[3] *= curr[1].buf[i];
//        u[1] *= curr[2].buf[i];
//        u[2] *= curr[3].buf[i];
//    }
////    }
//
//    return _mm_mul_ps(OMEGA_C,_mm_rcp14_ps(u));
//}

__m128 butterWorth_ps(__m128 u) {

    static const __m128 A = {0.390181f, 1.11114f, 1.66294f, 1.96157f};
    static const __m128 ONES = {1.f,1.f,1.f,1.f};
    static const __m128 OMEGA_C = {100.f,100.f,100.f,100.f};
//    static const __m128 OMEGA_C = {0.0008f,0.0008f,0.0008f,0.0008f};

    __m128 curr[4], squared;
    size_t i, j;

    for (i = 0; i < 4; ++i) {
        curr[i] = _mm_broadcast_ss(&(u[i]));
        squared = _mm_mul_ps(curr[i],curr[i]);
        curr[i] = _mm_mul_ps(A, curr[i]);
        curr[i] = _mm_add_ps(curr[i], squared);
        curr[i] = _mm_add_ps(ONES, curr[i]);
        curr[i] = _mm_mul_ps(OMEGA_C, curr[i]);
    }

//    for (i = 0; i < 4; ++i) {
        u[0] = curr[0][0];
        u[1] = curr[1][0];
        u[2] = curr[2][0];
        u[3] = curr[3][0];
        for (j = 1; j < 4; ++j) {
            u[0] *= curr[0][j];
            u[3] *= curr[1][j];
            u[1] *= curr[2][j];
            u[2] *= curr[3][j];
        }
//    }

    return _mm_mul_ps(OMEGA_C, _mm_rcp14_ps(u));
//    return _mm_rcp14_ps(u);
}

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

//    static const __m128 gainz = {1000000.f,1000000.f,1000000.f,1000000.f};
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

//        result = fmDemod(decimate(hComplexMultiply(shiftOrigin(*(__m256i *) (buf)))));

        for (i = 0; i < DEFAULT_BUF_SIZE; i += 32) {
//            result = _mm_mul_ps(oneHalf, _mm_hadd_ps(result,
//                  fmDemod(decimate(hComplexMultiply(shiftOrigin(*(__m256i *) (buf + i)))))));
            result = fmDemod(decimate(hComplexMultiply(shiftOrigin(*(__m256i *) (buf + i)))));
//            result = butterWorth_ps(result);
//            result = _mm_mul_ps(gainz, result);
            fwrite(&result, sizeof(__m128), 1, args->outFile);
        }
    }

    _mm_free(buf);
    return NULL;
}

void allocateBuffer(void **buf, const size_t len) {

    *buf = _mm_malloc(len, ALIGNMENT);
}
