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

    __m256 uhi, ulo,
            vhi, vlo;
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
//    const __m256i indexDC = _mm256_setr_epi8(
//            0, 3, 2, 1, 4, 7, 6, 5,
//            8, 11, 10, 9, 12, 15, 14, 13,
//            16, 19, 18, 17, 20, 23, 22, 21,
//            24, 27, 26, 25, 28, 31, 30, 29
////            0, 3, 1, 2, 4, 7, 5, 6,
////            8, 11, 9, 10, 12, 15, 13, 14,
////            16, 19, 17, 18, 20, 23, 21, 22,
////            24, 27, 25, 26, 28, 31, 29, 30
//    );

//    const __m256i indexCD = _mm256_setr_epi8(
//            2, 3, 0, 1, 6, 7, 4, 5,
//            10, 11, 8, 9, 14, 15, 12, 13,
//            18, 19, 16, 17, 22, 23, 20, 21,
//            26, 27, 24, 25, 30, 31, 28, 29
//    );
//    const __m256i indexAB = _mm256_setr_epi8(
//            0, 1, 4, 5, 2, 3, 6, 7,
//            8, 9, 12, 13, 10, 11, 14, 15,
//            16, 17, 20, 21, 18, 19, 22, 23,
//            24, 25, 28, 29, 26, 27, 30, 31
//    );
    const __m256i indexDC = _mm256_setr_epi8(
            1,0,3,2,5,4,7,6,
            9,8, 11, 10, 13, 12, 15, 14,
            17, 16, 19, 18, 21, 20, 23, 22,
            25,24, 27, 26, 29, 28, 31, 30
    );

    const __m256i indexInterleaveRealAndImag = _mm256_setr_epi8(
            0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15,
            0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15
    );
//    static const __m256i indexComplexConjugate = {
//            (int64_t) 0xffff0001ffff0001,
//            (int64_t) 0xffff0001ffff0001,
//            (int64_t) 0xffff0001ffff0001,
//            (int64_t) 0xffff0001ffff0001
//    };
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
    // a(-d)+bc // 0, 3, 2, 1
    v = _mm256_shuffle_epi8(v, indexDC);
    v = _mm256_sign_epi8(v, indexComplexConjugate);
    zj = _mm256_dpwssd_epi32(
            _mm256_setzero_si256(),
            _mm256_cvtepi8_epi16(_mm256_castsi256_si128(v)),
            _mm256_cvtepi8_epi16(_mm256_castsi256_si128(w)));
    zj = _mm256_inserti128_si256(zj,
            _mm256_castsi256_si128(
                    _mm256_dpwssd_epi32(_mm256_setzero_si256(),
                            _mm256_cvtepi8_epi16(_mm256_extracti128_si256(v, 1)),
                            _mm256_cvtepi8_epi16(_mm256_extracti128_si256(w, 1)))), 1);

//    v = _mm256_setr_m128i(          // c,d,...
//            _mm256_castsi256_si128(_mm256_cvtepi8_epi16(_mm256_castsi256_si128(v))),
//            _mm256_castsi256_si128(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(v, 1))));
//    w = _mm256_setr_m128i(          // a,b,...
//            _mm256_castsi256_si128(_mm256_cvtepi8_epi16(_mm256_castsi256_si128(w))),
//            _mm256_castsi256_si128(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(w, 1))));
//    zr = _mm256_mullo_epi16(w, v);  // ac, bd
//    zr = _mm256_hadd_epi16(zr, zr); // ac-(-b)d = ac+bd

//    v = _mm256_shuffle_epi8(_mm256_sign_epi16(v, indexComplexConjugate), indexDC); // -d,c,...
//    zj = _mm256_mullo_epi16(w, v);  // a(-d),bc
//    zj = _mm256_hadd_epi16(zj, zj); // a(-d)+bc
    return _mm256_shuffle_epi8(_mm256_blend_epi16(zr, zj, 0b11110000), indexInterleaveRealAndImag);
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
//    u_norm = _mm256_div_ps(u, hypot);
    // 64*y/(23*x*41*hypot)
    u = _mm256_mul_ps(_mm256_mul_ps(all64s, u), v);
//    v = _mm256_div_ps(_mm256_mul_ps(all64s, u_norm),
//          _mm256_permute_ps(_mm256_fmadd_ps(all23s, u_norm, all41s), _MM_SHUFFLE(2, 3, 0, 1)));

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
//    __m128 oneHalf = _mm_set1_ps(0.5f);

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
            fwrite(&result, sizeof(__m128), 1, args->outFile);
        }
    }

    _mm_free(buf);
    return NULL;
}

void allocateBuffer(void **buf, const size_t len) {

    *buf = _mm_malloc(len, ALIGNMENT);
}
