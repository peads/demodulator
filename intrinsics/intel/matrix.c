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

    return _mm256_add_epi8(u, ORIGIN_SHIFT_UINT8);
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

static inline __m256 filterButterWorth(__m256 u, const __m256 wc) {

    __m256 v;
    __m256 acc = {1, 1, 1, 1, 1, 1, 1, 1};
    size_t i;

    if (*(float *) &wc != 1) {
        v = _mm256_mul_ps(u, _mm256_rcp_ps(wc));
    } else {
        v = _mm256_rcp_ps(u);
    }

    for (i = 0; i < 8; ++i) {
        acc = _mm256_mul_ps(acc, _mm256_add_ps(v, BW_CONSTS[i]));
    }
    return _mm256_mul_ps(u, _mm256_rcp_ps(acc));
}

static inline __m256 filterRealButterworth(__m256 u, const __m256 wc) {

    size_t i;
    __m256 acc = ONES, v = _mm256_mul_ps(u, _mm256_rcp_ps(wc)),
        squared = _mm256_mul_ps(v, v);
    for (i = 0; i < 8; ++i) {
        acc = _mm256_mul_ps(acc, _mm256_add_ps(ONES,
                _mm256_add_ps(squared, _mm256_add_ps(v, BW_CONSTS_REAL[i]))));
    }
    return _mm256_mul_ps(u, _mm256_rcp_ps(acc));
}

static inline __m256 hComplexMulByConj(__m256 u) {

    __m256 temp = _mm256_mul_ps(
            _mm256_permute_ps(u, _MM_SHUFFLE(1, 0, 1, 0)),              // abab
            _mm256_mul_ps(_mm256_permute_ps(u, _MM_SHUFFLE(2, 3, 3, 2)),// cd(-d)c
                    MUL_CONJ));
    return _mm256_permutevar8x32_ps(_mm256_add_ps(temp,
            _mm256_permute_ps(temp, _MM_SHUFFLE(2, 3, 0, 1))), INDEX_CONJ_ORDERING);
}

static inline __m256 hPolarDiscriminant_ps(__m256 u, __m256 v) {

    v = hComplexMulByConj(v);
    return _mm256_blend_ps(hComplexMulByConj(u), _mm256_permute2f128_ps(v, v, 1), 0b11110000);
}

static inline __m256 fmDemod(__m256 u) {

    // fast atan2(y,x) := 64y/(23x+41*Sqrt[x^2+y^2])
    __m256 v = _mm256_mul_ps(u, u),
            hypot = _mm256_permute_ps(v, _MM_SHUFFLE(2, 3, 0, 1));
    hypot = _mm256_add_ps(v, hypot);
    hypot = _mm256_sqrt_ps(hypot);

    // 64*y
    v = _mm256_fmadd_ps(ALL_23S, u, _mm256_mul_ps(ALL_41S, hypot));
    // 1/(23*x+41*hypot)
    v = _mm256_permute_ps(_mm256_rcp_ps(v), _MM_SHUFFLE(2, 3, 0, 1));
    // 64*y/(23*x*41*hypot)
    u = _mm256_mul_ps(_mm256_mul_ps(ALL_64S, u), v);

    // NAN check
    return _mm256_permutevar8x32_ps(_mm256_and_ps(u, _mm256_cmp_ps(u, u, 0)), INDEX_FM_DEMOD_ORDERING);
}

void *processMatrix(void *ctx) {

    consumerArgs *args = ctx;
    size_t i;
    __m256 lowpassWc = _mm256_set1_ps(args->lowpassIn);
    __m256 highpassWc = _mm256_set1_ps(1.f/args->highpassIn);
    __m256 lowpassOutWc = _mm256_set1_ps(args->lowpassOut);
    __m256 result = {};
    __m256 hBuf[4] = {};
    __m256i u;
    uint8_t *buf = _mm_malloc(DEFAULT_BUF_SIZE, ALIGNMENT);

    while (!args->exitFlag) {
        sem_wait(args->full);
        pthread_mutex_lock(&args->mutex);
        memcpy(buf, args->buf, DEFAULT_BUF_SIZE);
        pthread_mutex_unlock(&args->mutex);
        sem_post(args->empty);

        for (i = 0; i < DEFAULT_BUF_SIZE; i += 32) {
            u = shiftOrigin(*(__m256i *) (buf + i));
            convert_epi8_ps(u, &hBuf[1], &hBuf[0], &hBuf[3], &hBuf[2]);

            if (args->lowpassIn) {
                hBuf[0] = filterButterWorth(hBuf[0], lowpassWc);
                hBuf[1] = filterButterWorth(hBuf[1], lowpassWc);
                hBuf[2] = filterButterWorth(hBuf[2], lowpassWc);
                hBuf[3] = filterButterWorth(hBuf[3], lowpassWc);
            }

            if (args->highpassIn) {
                hBuf[0] = filterButterWorth(hBuf[0], highpassWc);
                hBuf[1] = filterButterWorth(hBuf[1], highpassWc);
                hBuf[2] = filterButterWorth(hBuf[2], highpassWc);
                hBuf[3] = filterButterWorth(hBuf[3], highpassWc);
            }

            hBuf[0] = hPolarDiscriminant_ps(hBuf[0], hBuf[1]);
            hBuf[0] = fmDemod(hBuf[0]);

            hBuf[2] = hPolarDiscriminant_ps(hBuf[2], hBuf[3]);
            hBuf[1] = fmDemod(hBuf[2]);

            result = _mm256_blend_ps(hBuf[0],
                    _mm256_permute2f128_ps(hBuf[1], hBuf[1], 1),
                    0b11110000);

            if (args->lowpassOut) {
                result = filterRealButterworth(result, lowpassOutWc);
            }

            fwrite(&result, sizeof(__m256), 1, args->outFile);
        }
    }

    _mm_free(buf);
    return NULL;
}

void allocateBuffer(void **buf, const size_t len) {

    *buf = _mm_malloc(len, ALIGNMENT);
}
