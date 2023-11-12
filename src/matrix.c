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

static inline void fmDemod(const float *__restrict__ in,
                           const size_t len,
                           float *__restrict__ out) {

    size_t i;
    float zr, zj;

    for (i = 0; i < len; i += 4) {

        // ac-b(-d)=ac+bd
        // a(-d)+bc=-ad+bc
        zr = in[i] * in[i + 2] + in[i + 1] * in[i + 3];
        zj = -in[i] * in[i + 3] + in[i + 1] * in[i + 2];

        zr = 64.f * zj * frcpf(23.f * zr + 41.f * hypotf(zr, zj));
        out[i >> 2] = isnan(zr) ? 0.f : zr;
    }
}

float scaleSumButterworthPoles(size_t n, float fs, float fc) {

    size_t k;
    const float theta = (float)M_PI * fc/fs;
    float result[2] = {};
    float w;
    float a;
    float d;
    float zr;
    float zj;

    for (k = 1; k <= n; ++k) {
        w = M_PI_2 * (1.f / (float) n * (-1.f + (float) (k << 1)) + 1.f);
        a = cosf(w);
        d = -a + 1.f / sinf(2.f * theta);
        zr = 2.f + cosf(w)/d - 1.f/tanf(theta)/d;
        zj = -sinf(w) / d;
        if (k > 1) {
            result[0] = zr * result[0] - zj * result[1];
            result[1] = zr * result[1] + zj * result[0];
        } else {
            result[0] = zr;
            result[1] = zj;
        }

        fprintf(stderr, "%f + I %f, ", zr, zj);
    }

    fprintf(stderr, "\n");
    return result[0];
}

static inline void shiftOrigin(void *__restrict__ in, const size_t len, float *__restrict__ out) {

    size_t i;
    int8_t *buf = in;
    for (i = 0; i < len; i += 2) {
        out[i] = (int8_t) (buf[i] - 127);
        out[i + 1] = (int8_t) (buf[i + 1] - 127);
    }
}

static inline void balanceIq(float *__restrict__ buf, size_t len) {

    static const float alpha = 0.99212598425f;
    static const float beta = 0.00787401574f;

    size_t i;
    for (i = 0; i < len; i += 2) {
        buf[i] *= alpha;
        buf[i + 1] += beta * buf[i];
    }
}

static inline void filterOut(float *__restrict__ x,
                             size_t len,
                             size_t filterLen,
                             float *__restrict__ y,
                             const float *__restrict__ coeffA,
                             const float *__restrict__ coeffB) {

    float *xp, *yp, acc;
    size_t i, j, k;

    for (i = 0; i < len; ++i) {
        xp = &x[3 + i];
        yp = &y[3 + i];
        acc = 0;
        for (j = 0; j < filterLen; ++j) {
            k = filterLen - j - 1;
            acc += coeffA[k] * xp[j] - coeffB[k] * yp[j];
        }
        y[i] = acc;
    }
}

void *processMatrix(void *ctx) {

    static const float coeffALow[] = {0.0005e-4f, 0.0063e-4f, 0.0377e-4f, 0.1384e-4f, 0.3460e-4f, 0.6227e-4f, 0.8303e-4f, 0.8303e-4f, 0.6227e-4f, 0.3460e-4f, 0.1384e-4f, 0.0377e-4f, 0.0063e-4f, 0.0005e-4f};
    static const float coeffBLow[] = {1.0000f, -7.5823f, 27.2800f, -61.4122f, 96.2248f, -110.5639f, 95.6695f, -63.0250f, 31.5708f, -11.8641f, 3.2480f, -0.6130f, 0.0714f, -0.0039f};
    static const float N = 14;
    size_t i = 3;
    consumerArgs *args = ctx;
    void *buf = calloc(DEFAULT_BUF_SIZE, 1);
    float *fBuf = calloc(DEFAULT_BUF_SIZE, sizeof(float));
    float *demodRet = calloc(DEFAULT_BUF_SIZE, sizeof(float));

    for (; i < 21; ++i) {
        fprintf(stderr, "\n%lu: %f\n", i+1, scaleSumButterworthPoles(i, 125.f, 13.f));
    }
    while (!args->exitFlag) {

        float *filterRet = calloc(DEFAULT_BUF_SIZE, sizeof(float));
        sem_wait(args->full);
        pthread_mutex_lock(&args->mutex);
        memcpy(buf, args->buf, DEFAULT_BUF_SIZE);
        pthread_mutex_unlock(&args->mutex);
        sem_post(args->empty);

        shiftOrigin(buf, DEFAULT_BUF_SIZE, fBuf);
        balanceIq(fBuf, DEFAULT_BUF_SIZE);
        fmDemod(fBuf, DEFAULT_BUF_SIZE, demodRet);
        filterOut(demodRet, DEFAULT_BUF_SIZE >> 2, N, filterRet, coeffALow, coeffBLow);
        fwrite(filterRet, sizeof(float), DEFAULT_BUF_SIZE >> 2, args->outFile);
        free(filterRet);
    }
    free(buf);
    free(fBuf);
    free(demodRet);

    return NULL;
}

void allocateBuffer(void **buf, const size_t len) {

    *buf = calloc(len, 1);
}
