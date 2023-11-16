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

typedef void (*poleGenerator_t)(size_t, size_t, float *, float *);
static float theta = (float) M_PI * 13.f / 125.f;

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

static inline void butter(const size_t k, const size_t n, float *acc, float *z) {

    size_t j = (k - 1) << 1;
    float w = M_PI_2 * (1.f / (float) n * (-1.f + (float) (k << 1)) + 1.f);
    float a = cosf(w);
    float d = 1.f / (a - 1.f / sinf(2.f * theta));
    float zr = (a - tanf(theta)) * d;
    float zj = sinf(w) * d;

    z[j] = 1.f - zr;
    z[j + 1] = zj;

    a = zr * acc[0] - zj * acc[1];
    acc[1] = zr * acc[1] + zj * acc[0];
    acc[0] = a;

    a = (2.f - zr) * acc[2] - zj * acc[3];
    acc[3] = (2.f - zr) * acc[3] + zj * acc[2];
    acc[2] = a;
}

static inline float transformBilinear(const size_t n,
                                      float *__restrict__ A,
                                      float *__restrict__ B,
                                      float *__restrict__ C,
                                      poleGenerator_t fn) {

    size_t i, j, k;
    float b = 1.f;
    float acc[4] = {1.f, 0, 1.f, 0};
    float *p = calloc(((n + 1) << 1), sizeof(float));
    float *z = calloc((n << 1), sizeof(float));
    float *t = calloc((n << 1), sizeof(float));
    p[0] = 1.f;
    B[0] = 1.f;

    // Generate roots of bilinear transform
    // Perform running sum of coefficients
    // Expand roots into coefficients of monic polynomial
    for (j = 0, k = 1; k <= n; j += 2, ++k) {

        B[k] = B[k - 1] * (float) (n - k + 1) / (float) (k);
        b += B[k];
        fn(k, n, acc, z);

        for (i = 0; i <= j; i += 2) {
            t[i] = z[j] * p[i] - z[j + 1] * p[i + 1];
            t[i + 1] = z[j] * p[i + 1] + z[j + 1] * p[i];
        }
        for (i = 0; i < j + 2; i += 2) {
            p[i + 2] -= t[i];
            p[i + 3] -= t[i + 1];
        }
    }

    // Store the output
    acc[0] /= b;
    acc[2] /= b;
    for (k = 0; k < n + 1; ++k) {
        C[k] = (!(k & 1) ? B[k] : -B[k]) * acc[2];
        B[k] *= acc[0];
        A[k] = p[k << 1];
    }
    free(p);
    free(t);
    free(z);
    return acc[0];
}

static inline void shiftOrigin(void *__restrict__ in, const size_t len, float *__restrict__ out) {

    size_t i;
    int8_t *buf = in;
    for (i = 0; i < len; i += 2) {
        out[i] = (int8_t) (buf[i] - 127);
        out[i + 1] = (int8_t) (buf[i + 1] - 127);
    }
}

static inline void balanceIq(float *__restrict__ buf, const size_t len) {

    static const float alpha = 0.99212598425f;
    static const float beta = 0.00787401574f;

    size_t i;
    for (i = 0; i < len; i += 2) {
        buf[i] *= alpha;
        buf[i + 1] += beta * buf[i];
    }
}

static inline void filterOut(float *__restrict__ x,
                             const size_t len,
                             const size_t filterLen,
                             float *__restrict__ y,
                             const float *__restrict__ A,
                             const float *__restrict__ B) {

    float *xp, *yp, acc;
    size_t i, j, k;

    for (i = 0; i < len; ++i) {
        xp = &x[filterLen + i];
        yp = &y[filterLen + i];
        acc = 0;
        for (j = 0; j < filterLen; ++j) {
            k = filterLen - j - 1;
            acc += A[k] * xp[j] - B[k] * yp[j];
        }
        y[i] = acc;
    }
}

void *processMatrix(void *ctx) {

    static const size_t filterLength = 7;
    float *A = calloc(filterLength + 1, sizeof(float));
    float *B = calloc(filterLength + 1, sizeof(float));
    float *C = calloc(filterLength + 1, sizeof(float));
    void *buf = calloc(DEFAULT_BUF_SIZE, 1);
    float *fBuf = calloc(DEFAULT_BUF_SIZE, sizeof(float));
    float *demodRet = calloc(DEFAULT_BUF_SIZE, sizeof(float));
    consumerArgs *args = ctx;

    theta = args->lowpassOut && args->sampleRate ? (float) M_PI * args->lowpassOut / args->sampleRate : theta;
    transformBilinear(filterLength, A, B, C, butter);

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
        filterOut(demodRet, DEFAULT_BUF_SIZE >> 2, filterLength + 1, filterRet, B, A);
        fwrite(filterRet, sizeof(float), DEFAULT_BUF_SIZE >> 2, args->outFile);
        free(filterRet);
    }
    free(buf);
    free(fBuf);
    free(demodRet);
    free(A);
    free(B);
    free(C);

    return NULL;
}

void allocateBuffer(void **buf, const size_t len) {

    *buf = calloc(len, 1);
}
