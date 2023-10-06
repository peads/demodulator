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
#include <arm_neon.h>
#include <stdlib.h>
#include "definitions.h"
#include "matrix.h"

static inline int8x16_t boxcarEpi8(int8x16_t u) {

    static const int8x16_t Z = {
            1,-1,1,-1,1,-1,1,-1,
            1,-1,1,-1,1,-1,1,-1
    };
    static const uint8x16_t mask = {
            2,3,0,1,6,7,4,5,
            10,11,8,9,14,15,12,13
    };

    u = vmulq_s8(u, Z);
    int8x16_t v = vqtbl1q_s8(u, mask);
    return vaddq_s8(u, v);
}

static int8x16_t convert_epu8_epi8(uint8x16_t u) {

    static const int8x16_t Z = {
            -127,-127,-127,-127,-127,-127,-127,-127,
            -127,-127,-127,-127,-127,-127,-127,-127
    };
    return vaddq_s8(vreinterpretq_s8_u8(u), Z);
}

static inline void demodEpi8(uint8x16_t buf, float *__restrict__ result) {

    static const int8x16_t negateBIm = {
            1,1,1,-1,1,1,1,-1,
            1,1,1,-1,1,1,1,-1
    };
    static const uint8x16_t indexLo = {
            0,1,2,3,4,5,2,3,
            4,5,6,7,8,9,6,7
    };
    static const uint8x16_t indexHi = {
            8, 9, 10,11,12,13,11,12,
            12,13,14,15,16,17,14,15
    };

    int8x16_t lo, hi, u = convert_epu8_epi8(buf);
    u = boxcarEpi8(u);
    hi = vqtbl1q_s8(u, indexHi);
    hi = vmulq_s8(hi, negateBIm);
    lo = vqtbl1q_s8(u, indexLo);
    lo = vmulq_s8(lo, negateBIm);
    hi=hi;
    lo=lo;
}

void *processMatrix(void *ctx) {

    size_t i, j;
    consumerArgs *args = ctx;
    void *buf __attribute__((aligned(16))) = calloc(DEFAULT_BUF_SIZE, 1);
    float result[DEFAULT_BUF_SIZE >> 4];

    while (!args->exitFlag) {
        sem_wait(&args->full);
        pthread_mutex_lock(&args->mutex);
        memcpy(buf, args->buf, DEFAULT_BUF_SIZE);
        pthread_mutex_unlock(&args->mutex);
        sem_post(&args->empty);

        for (i = 0, j = 0; i < DEFAULT_BUF_SIZE; i += 16, j += 4) {
            demodEpi8(*(uint8x16_t*) (buf + i), result + j);

            if (args->gain) {
                vmulq_n_f32(*(float32x4_t*)result, args->gain);
            }
        }
        fwrite(result, sizeof(float), DEFAULT_BUF_SIZE >> 3, args->outFile);
    }
    return NULL;
}

void allocateBuffer(void **buf, const size_t len) {

    *buf = calloc(len, 1);
}