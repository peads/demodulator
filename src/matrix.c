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

typedef void (*findFilterPolesFn_t)(size_t, size_t, double, double, double *);

static const double epsilon = 1e-15;

static inline void butterworthPole(size_t k,
                                   size_t n,
                                   double fs,
                                   double fc,
                                   double *result) {
    // Simplification of bilinear transform of Butterworth transfer fn
    // ((1 + Exp[(2 k + n - 1)/(2 n) Pi I] Tan[Pi fc/fs])
    // / (1 - Exp[(2 k + n - 1)/(2 n) Pi I] Tan[Pi fc/fs]))
    const double ratio = 2. * M_PI * fc / fs;
    const double ratio1 = M_PI_2 / (double) n * (1. - 2. * (double) k);
    const double SIN = sin(ratio);
    const double mag = SIN * sin(ratio1) - 1.;

    result[0] = -cos(ratio) / mag;
    result[1] = -cos(ratio1) * SIN / mag;

    result[0] = ((fabs(0 - result[0]) < epsilon) ? 0 : result[0]);
    result[1] = ((fabs(0 - result[1]) < epsilon) ? 0 : result[1]);
}

double polynomialExpand(size_t len,
                      double fs,
                      double fc,
                      findFilterPolesFn_t fn,
                      double *roots) {

    size_t i;

    // already negated for ease
    for (i = 2; i <= len; i+=2) {
        fn(i >> 1, len >> 1, fs, fc, &roots[len - i]);
        roots[len - i] = 1 - roots[len - i];
        roots[len - i + 1] = -roots[len - i + 1];
    }
//    double ac;
//    double bd;
    for (i = 2; i < len; i+=2) {
//        ac = roots[0] * roots[len - i];
//        bd = roots[1] * roots[len - i + 1];
//        roots[0] = ac - bd;
//        roots[1] = (roots[0] + roots[1])*(roots[len - i] + roots[len - i + 1]) - ac - bd;
        roots[0] = roots[0] * roots[len - i] - roots[1] * roots[len - i + 1];
        roots[1] = roots[0] * roots[len - i + 1] + roots[1] * roots[len - i];
    }

    return roots[0];
}

float mapFilterAnalogToDigitalDomain(size_t len, double fc, double fs, findFilterPolesFn_t fn) {

    const size_t n = (len >> 1) + 1;
    const size_t NP1 = len + 1;
    double coeffB[n];
    double *roots = calloc(len << 1, sizeof(double));
    size_t i, j;
    double sum = 0;

    for (i = 0, j = 0; i < n; ++i, j += 2) {
        coeffB[i] = round(exp(
                lgamma((double) NP1)
                - lgamma((double) (i + 1))
                - lgamma((double) (NP1 - i))));
        sum += coeffB[i];
    }
    sum *= 2.;
    sum /= polynomialExpand(len << 1, fs, fc, fn, roots);
    free(roots);

    return sum;
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
            mapFilterAnalogToDigitalDomain(5, 1.3e4, 1.25e5, butterworthPole));
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
