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
static int8x16_t convert_epu8_epi8(uint8x16_t u) {
    static const int8_t onetwoseven = -127;

    const int8x16_t Z = vld1q_dup_s8(&onetwoseven);
    return vaddq_s8(vreinterpretq_s8_u8(u), Z);
}

static inline void demodEpi8(uint8x16_t u, float *__restrict__ result) {
    fprintf(stderr, "Starting demodulation");
//    uint8x16_t negateBIm = 0xff010ff0101;

    convert_epu8_epi8(u);
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