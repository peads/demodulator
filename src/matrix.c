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

typedef void (*findFilterPolesFn_t)(size_t, size_t, float, float, void *);

static const float epsilon = 1e-6f;

void cAdd(void *a, const void *b, const void *c) {

    float *dest = a;
    const float *src1 = b;
    const float *src2 = c;
    dest[0] = src1[0] + src2[0];
    dest[1] = src1[1] + src2[1];

    dest[0] = ((fabsf(0 - dest[0]) < epsilon) ? 0 : dest[0]);
    dest[1] = ((fabsf(0 - dest[1]) < epsilon) ? 0 : dest[1]);
}

void cMul(void *a, void *b, void *c) {

    float *dest = a;
    const float *src1 = b;
    const float *src2 = c;

    dest[0] = src1[0] * src2[0] - src1[1] * src2[1];
    dest[1] = src1[0] * src2[1] + src1[1] * src2[0];
}

static inline void butterworthPole(size_t k,
                                   size_t n,
                                   float fs,
                                   float fc,
                                   void *out) {
    // Simplification of bilinear transform of Butterworth transfer fn
    // ((1 + Exp[(2 k + n - 1)/(2 n) Pi I] Tan[Pi fc/fs])
    // / (1 - Exp[(2 k + n - 1)/(2 n) Pi I] Tan[Pi fc/fs]))
    const float ratio = M_PI * 2.f * fc / fs;
    const float ratio1 = M_PI_2 / (float) n * (1 - 2.f * (float) k);
    const float SIN = sinf(ratio);
    const float mag = SIN * sinf(ratio1) - 1.f;

    float *result = out;
    result[0] = -cosf(ratio) / mag;
    result[1] = -cosf(ratio1) * SIN / mag;

    result[0] = 1.f + ((fabsf(0 - result[0]) < epsilon) ? 0 : -result[0]);
    result[1] = -((fabsf(0 - result[1]) < epsilon) ? 0 : -result[1]);
}

void polynomialExpand(size_t len,
                      float fs,
                      float fc,
                      findFilterPolesFn_t fn,
                      void *out) {

    double *roots = out;
    size_t i;

    // already negated for ease
    for (i = 1; i <= len; ++i) {
        fn(i, len, fs, fc, &roots[len - i]);
    }

    for (i = 1; i < len; ++i) {
        cMul(&roots[0], &roots[0], &roots[len - i]);
    }
}

float mapFilterAnalogToDigitalDomain(size_t len, float fc, float fs, findFilterPolesFn_t fn) {

    const size_t n = (len >> 1) + 1;
    const size_t NP1 = len + 1;
    float coeffB[n];
    float *roots = calloc(NP1, sizeof(float));
    size_t i, j;
    float sumB = 0;

    for (i = 0, j = 0; i < n; ++i, j += 2) {
        coeffB[i] = roundf(expf(
                lgammaf((float) NP1)
                - lgammaf((float) (i + 1))
                - lgammaf((float) (NP1 - i))));
        sumB += coeffB[i];
    }
    polynomialExpand(len, fs, fc, fn, roots);
    free(roots);
    return sumB * 2.f;
}

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

    static const float coeffALow[] = {0.0034f, 0.0236f, 0.0707f, 0.1178f, 0.1178f, 0.0707f, 0.0236f, 0.0034f};
    static const float coeffBLow[] = {1.0000f, -1.8072f, 2.3064f, -1.7368f, 0.9219f, -0.3115f, 0.0641f, -0.0060f};
    consumerArgs *args = ctx;
    void *buf = calloc(DEFAULT_BUF_SIZE, 1);
    float *fBuf = calloc(DEFAULT_BUF_SIZE, sizeof(float));
    float *demodRet = calloc(DEFAULT_BUF_SIZE, sizeof(float));
    fprintf(stderr, "\nSum: %f\n",
            mapFilterAnalogToDigitalDomain(12, 1.3e4f, 1.25e5f, butterworthPole));
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
        filterOut(demodRet, DEFAULT_BUF_SIZE >> 2, 8, filterRet, coeffALow, coeffBLow);
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
