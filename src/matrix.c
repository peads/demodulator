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
#include <stdlib.h>
#include "matrix.h"
#include "fmath.h"

static inline void fmDemod(const void *__restrict__ in,
                           const uint32_t len,
                           const float gain,
                           float *__restrict__ result) {

    float temp[4] = {0.f, 0.f, 0.f, 0.f};
    const uint8_t *buf = in;
    uint32_t i;
    float zr, zj, ac, bd;

    for (i = 0; i < len; i += 2) {

        temp[0] = (float) (buf[i] + buf[i + 2] - 254);       // ar
        temp[1] = (float) (buf[i + 1] + buf[i + 3] - 254);   // aj
        temp[2] = (float) (buf[i + 4] + buf[i + 6] - 254);   // br
        temp[3] = (float) (254 - buf[i + 5] - buf[i + 7]);   // -bj

        ac = temp[0] * temp[2];
        bd = temp[1] * temp[3];
        zr = ac - bd;
        zj = (temp[0] + temp[1]) * (temp[2] + temp[3]) - (ac + bd);
        zr = 64.f * zj * frcpf(23.f * zr + 41.f * hypotf(zr, zj));

        result[i >> 2] = isnan(zr) ? 0.f : gain ? zr * gain : zr;
    }
}

void *processMatrix(void *ctx) {

    consumerArgs *args = ctx;
    void *buf = calloc(DEFAULT_BUF_SIZE, 1);
    float *result = calloc(DEFAULT_BUF_SIZE >> 2, sizeof(float));

    while (!args->exitFlag) {

        sem_wait(&args->full);
        pthread_mutex_lock(&args->mutex);
        memcpy(buf, args->buf, DEFAULT_BUF_SIZE);
        pthread_mutex_unlock(&args->mutex);
        sem_post(&args->empty);

        fmDemod(buf, DEFAULT_BUF_SIZE, args->gain, result);
        fwrite(result, sizeof(float), DEFAULT_BUF_SIZE >> 2, args->outFile);
    }
    free(buf);
    free(result);

    return NULL;
}

void allocateBuffer(void **buf, const size_t len) {

    *buf = calloc(len, 1);
}
