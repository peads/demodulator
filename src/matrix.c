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
#include "filter.h"

static inline float generateHannCoefficient(const size_t k, const size_t n) {
//static double *windowIn = NULL;
    static float *windowOut = NULL;
    if (!windowOut) {
        size_t i, N = n >> 1;
        N = (n & 1) ? N + 1 : N;
        windowOut = calloc(n, sizeof(float));
        double x;
        for (i = 0; i < N; ++i) {
            x = sin(M_PI * (double) i / (double) n);
            windowOut[n - i - 1] = windowOut[i] = (float) (x * x);
        }
    }
    return windowOut[k];
}

static inline void shiftOrigin(
        void *__restrict__ in,
        const size_t len,
        float *__restrict__ out) {

    size_t i;
    int8_t *buf = in;
    for (i = 0; i < len >> 1; i += 2) {
        out[i] = (int8_t) (buf[i] - 127);
        out[i + 1] = (int8_t) (buf[i + 1] - 127);

        out[len - i - 2] = (int8_t) (buf[len - i - 2] - 127);
        out[len - i - 1] = (int8_t) (buf[len - i - 1] - 127);
    }
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

        zr = 64.f * zj * 1.f / (23.f * zr + 41.f * hypotf(zr, zj));
        out[i >> 2] = isnan(zr) ? 0.f : zr;
    }
}

static inline void processFilterOption(uint8_t mode,
                                       size_t degree,
                                       float sosf[][6],
                                       double fc,
                                       double fs,
                                       double epsilon) {

    size_t N = degree >> 1;
    N = (degree & 1) ? N + 1 : N;
    const double w = M_PI * fc / fs;
    size_t i, j;
    double sos[N][6];
    double wh;

    if (mode) {
        wh = cosh(1. / (double) degree * acosh(1. / sqrt(pow(10., epsilon) - 1.)));
#ifdef VERBOSE
        fprintf(stderr, "\nepsilon: %f\nwc: %f", epsilon * 10., wh * fc);
#endif
        wh =  tan(wh * w);
        transformBilinear(degree, wh, epsilon, sos, warpCheby1);
    } else {
        transformBilinear(degree, 1./sin(2. * w), tan(w), sos, warpButter);
    }

    for (i = 0; i < N; ++i) {
        for (j = 0; j < 6; ++j) {
            sosf[i][j] = (float) sos[i][j];
        }
    }
}

void *processMatrix(void *ctx) {

    float *filterRet;
    consumerArgs *args = ctx;
    void *buf = calloc(args->bufSize, 1);
    float *fBuf = calloc(args->bufSize, sizeof(float));
    float *demodRet = calloc(args->bufSize, sizeof(float));
    size_t filterOutputLength = args->bufSize;

    args->sampleRate = args->sampleRate ? args->sampleRate : 10.f;
    args->lowpassOut = args->lowpassOut ? args->lowpassOut : 1.f;
    args->outFilterDegree = args->outFilterDegree ? args->outFilterDegree : 7;
    args->inFilterDegree = args->inFilterDegree ? args->inFilterDegree : 7;
    args->epsilon = args->epsilon ? args->epsilon : .3f;

    size_t sosLen = args->outFilterDegree >> 1;
    sosLen = (args->outFilterDegree & 1) ? sosLen + 1 : sosLen;
    float sosIn[sosLen][6];
    float sosOut[sosLen][6];

    if (!args->lowpassIn) {
        processFilterOption(args->mode & 1,
                args->outFilterDegree, sosOut, args->lowpassOut, args->sampleRate, args->epsilon);
    } else {
        processFilterOption(args->mode & 1,
                args->outFilterDegree, sosOut, args->lowpassOut, args->sampleRate, args->epsilon);
        filterOutputLength <<= 1;
        processFilterOption((args->mode >> 1) & 1,
                args->outFilterDegree, sosIn, args->lowpassIn, args->sampleRate, args->epsilon);
    }

    filterRet = calloc(filterOutputLength, sizeof(float));

    while (!args->exitFlag) {

        sem_wait(args->full);
        pthread_mutex_lock(&args->mutex);
        memcpy(buf, args->buf, args->bufSize);
        pthread_mutex_unlock(&args->mutex);
        sem_post(args->empty);

        shiftOrigin(buf, args->bufSize, fBuf);
        if (args->demodMode && !args->lowpassIn) {
            fmDemod(fBuf, args->bufSize, demodRet);
            applyFilter(demodRet, filterRet, args->bufSize >> 2,
                    sosLen, sosOut, generateHannCoefficient);
            fwrite(filterRet, sizeof(float), args->bufSize >> 2, args->outFile);
        } else {
            if (!args->demodMode) {
                applyComplexFilter(fBuf, filterRet, args->bufSize, sosLen, sosIn, generateHannCoefficient);
                fwrite(filterRet, sizeof(float), args->bufSize, args->outFile);
            } else {
                applyComplexFilter(fBuf, filterRet, args->bufSize, sosLen, sosIn, generateHannCoefficient);
                fmDemod(filterRet, args->bufSize, demodRet);
                applyFilter(demodRet, filterRet + args->bufSize, args->bufSize >> 2, sosLen, sosOut, generateHannCoefficient);
                fwrite(filterRet + args->bufSize, sizeof(float),
                        args->bufSize >> 2, args->outFile);
            }
        }
        memset(filterRet, 0, filterOutputLength * sizeof(float));
    }
    free(buf);
    free(fBuf);
    free(demodRet);

    return NULL;
}

void allocateBuffer(void **buf, const size_t len) {

    *buf = calloc(len, 1);
}
