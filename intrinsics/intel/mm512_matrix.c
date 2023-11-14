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
#include <math.h>

static inline __m512i shiftOrigin(__m512i u) {

    return _mm512_add_epi8(u, ORIGIN_SHIFT_UINT8);
}

static inline __m512 hComplexMulByConj(__m512 u) {
    //TODO make static const in header
    const __m512i indexOrdering =
            _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);

    __m512 temp = _mm512_mul_ps(
            _mm512_permute_ps(u, _MM_SHUFFLE(1, 0, 1, 0)),              // abab
            _mm512_mul_ps(_mm512_permute_ps(u, _MM_SHUFFLE(2, 3, 3, 2)),// cd(-d)c
                    MUL_CONJ));
    return _mm512_permutexvar_ps(indexOrdering, _mm512_add_ps(temp,
            _mm512_permute_ps(temp, _MM_SHUFFLE(2, 3, 0, 1))));
}

static inline __m512 hPolarDiscriminant_ps(__m512 u, __m512 v) {

    //TODO make static const in header
    const __m512i index = _mm512_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7);

    v = hComplexMulByConj(v);
    return _mm512_mask_blend_ps(0b1111111100000000,
            hComplexMulByConj(u), _mm512_permutexvar_ps(index, v));
}

static inline __m512 fmDemod(__m512 u) {

    //TODO make static const in header
    const __m512i INDEX_FM_DEMOD_ORDERING = _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 0, 2, 3, 6, 8, 10, 12, 14);
    // fast atan2(y,x) := 64y/(23x+41*Sqrt[x^2+y^2])
    __m512 v = _mm512_mul_ps(u, u),
            hypot = _mm512_permute_ps(v, _MM_SHUFFLE(2, 3, 0, 1));
    hypot = _mm512_add_ps(v, hypot);
    hypot = _mm512_sqrt_ps(hypot);

    // 64*y
    v = _mm512_fmadd_ps(ALL_23S, u, _mm512_mul_ps(ALL_41S, hypot));
    // 1/(23*x+41*hypot)
    v = _mm512_permute_ps(_mm512_rcp14_ps(v), _MM_SHUFFLE(2, 3, 0, 1));
    // 64*y/(23*x*41*hypot)
    u = _mm512_mul_ps(_mm512_mul_ps(ALL_64S, u), v);

    // NAN check
    u = _mm512_maskz_and_ps(_mm512_cmp_ps_mask(u, u, 0), u, u);
    u = _mm512_permutexvar_ps(INDEX_FM_DEMOD_ORDERING, u);
    return u;
}

static inline __m512 balanceIq(__m512 u) {
    
    __m512 ia = _mm512_mul_ps(u, alpha);
    __m512 ib = _mm512_mul_ps(beta, _mm512_permute_ps(ia, _MM_SHUFFLE(2,3,0,1)));
    return _mm512_add_ps(ia, ib);
}

void *processMatrix(void *ctx) {

    // TODO convert to header static const
    const __m512i index = _mm512_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7);

    consumerArgs *args = ctx;
    size_t i;
    __m512 result = {};
    __m512 hBuf[4] = {};
    __m512i u, v;
    uint8_t *buf = _mm_malloc(DEFAULT_BUF_SIZE, ALIGNMENT);

    while (!args->exitFlag) {
        sem_wait(args->full);
        pthread_mutex_lock(&args->mutex);
        memcpy(buf, args->buf, DEFAULT_BUF_SIZE);
        pthread_mutex_unlock(&args->mutex);
        sem_post(args->empty);

        for (i = 0; i < DEFAULT_BUF_SIZE; i += 64) {
            u = shiftOrigin(*(__m512i *) (buf + i));

            v = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(u));
            hBuf[0] = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(v)));
            hBuf[1] = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(v, 1)));

            hBuf[0] = balanceIq(hBuf[0]);
            hBuf[1] = balanceIq(hBuf[1]);

            v = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(u,1));
            hBuf[2] = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(v)));
            hBuf[3] = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(v, 1)));

            hBuf[2] = balanceIq(hBuf[2]);
            hBuf[3] = balanceIq(hBuf[3]);

            hBuf[0] = hPolarDiscriminant_ps(hBuf[0], hBuf[1]);
            hBuf[0] = fmDemod(hBuf[0]);

            hBuf[2] = hPolarDiscriminant_ps(hBuf[2], hBuf[3]);
            hBuf[1] = fmDemod(hBuf[2]);

            result = _mm512_mask_blend_ps(0b1111111100000000, hBuf[0],
                    _mm512_permutexvar_ps(index, hBuf[1]));

            fwrite(&result, sizeof(__m512), 1, args->outFile);
        }
    }

    _mm_free(buf);
    return NULL;
}

void allocateBuffer(void **buf, const size_t len) {

    *buf = _mm_malloc(len, ALIGNMENT);
}
