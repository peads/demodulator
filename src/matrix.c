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

static inline void generateHannCoefficient(const size_t n, REAL *__restrict__ wind) {

    size_t i, N = n >> 1;
    LREAL x;
    N = (n&1)?N+1:N;
    for (i = 0; i < N; ++i) {
        x = SIN(M_PI * (LREAL) i / (LREAL) n);
        wind[n - i - 1] =
        wind[i] = (REAL) (x * x);
#ifdef VERBOSE
        fprintf(stderr, "%f ", wind[i]);
#endif
    }
    fprintf(stderr, "\n");
}

static inline void processFilterOption(uint8_t mode,
                                       size_t degree,
                                       float sosf[][6],
                                       LREAL fc,
                                       LREAL fs,
                                       LREAL epsilon) {

    size_t N = degree >> 1;
    N = (degree & 1) ? N + 1 : N;
    const LREAL w = M_PI * fc / fs;
    size_t i, j;
    LREAL sos[N][6];
    LREAL wh;

#ifdef VERBOSE
    fprintf(stderr, "\ndegree: %zu", degree);
#endif

    if (!mode) {
        transformBilinear(degree, 1. / SIN(2. * w), TAN(w), sos, warpButter);
    } else {
        wh = COSH(1. / (LREAL) degree * ACOSH(1. / SQRT(POW(10., epsilon) - 1.)));
#ifdef VERBOSE
        fprintf(stderr, PRINT_EP_WC, epsilon * 10., wh * fc);
#endif
        wh = TAN(wh * w);
        transformBilinear(degree, wh, epsilon, sos, warpCheby1);
    }

    for (i = 0; i < N; ++i) {
        for (j = 0; j < 6; ++j) {
            sosf[i][j] = (float) sos[i][j];
        }
    }
}

static inline void shiftOrigin(
        void *__restrict__ in,
        const size_t len,
        float *__restrict__ out) {

    const size_t N = len >> 1;
    size_t i;
    uint8_t *buf = in;

    for (i = 0; i < N; i += 2) {
        out[i] = (int8_t) (buf[i] - 127);
        out[i + 1] = (int8_t) (buf[i + 1] - 127);

        out[len - i - 2] = (int8_t) (buf[len - i - 2] - 127);
        out[len - i - 1] = (int8_t) (buf[len - i - 1] - 127);
    }
}

static inline void convertU8ToReal (
        void *__restrict__ in,
        const size_t len,
        float *__restrict__ out) {
    const size_t N = len >> 1;
    size_t i;
    uint8_t *buf = in;

    for (i = 0; i < N; i += 2) {
        out[i] = (float) buf[i];
        out[i + 1] = (float) buf[i + 1];

        out[len - i - 2] = (float) buf[len - i - 2];
        out[len - i - 1] = (float) buf[len - i - 1];
    }
}

static float esr;
static inline void correctIq(
        void *__restrict__ in,
        const size_t len,
        float *__restrict__ out) {

    static float off[2] = {};
    const size_t N = len >> 1;
    size_t i;
    uint8_t *buf = in;

    for (i = 0; i < N; i += 2) {
        out[i] = ((float) buf[i]) - off[0];
        out[len - i - 2] = ((float) buf[len - i - 2]) - off[0];

        out[i+1] = ((float) buf[i+1]) - off[1];
        out[len - i - 1] = ((float) buf[len - i - 1]) - off[1];

        off[0] += (out[i] + out[len - i - 2]) * esr;
        off[1] += (out[i+1] + out[len - i - 1]) * esr;
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

void *processMatrix(void *ctx) {

    consumerArgs *args = ctx;
    void *buf = calloc(args->bufSize, 1);

    float *filterRet;
    float *fBuf = calloc(args->bufSize, sizeof(float));
    float *demodRet = calloc(args->bufSize, sizeof(float));

    size_t filterOutputLength = args->bufSize;

    const size_t outputLen = args->bufSize >> 2;
    const uint8_t iqMode = (args->mode >> 2) & 1;
    const uint8_t demodMode = (args->mode >> 4) & 3;
    const iqCorrection_t processInput =
            (args->mode >> 3) & 1 ? convertU8ToReal
                : (iqMode ? shiftOrigin
                    : correctIq);
    const size_t sosLen =
            (args->outFilterDegree & 1) ? (args->outFilterDegree >> 1) + 1
                : (args->outFilterDegree >> 1);

    float sosIn[sosLen][6];
    float sosOut[sosLen][6];
    float *windIn = NULL;
    float *windOut = calloc(args->outFilterDegree, sizeof(float));

    generateHannCoefficient(args->outFilterDegree, windOut);
    if (!args->lowpassIn) {
        processFilterOption(args->mode & 1,
                args->outFilterDegree, sosOut, args->lowpassOut, args->sampleRate, args->epsilon);
    } else {
        args->inFilterDegree = args->outFilterDegree; // TODO decouple out and in filter lens
        windIn = calloc(args->inFilterDegree, sizeof(float));
        generateHannCoefficient(args->inFilterDegree, windIn);
        processFilterOption(args->mode & 1,
                args->outFilterDegree, sosOut, args->lowpassOut, args->sampleRate, args->epsilon);
        filterOutputLength <<= 1;
        processFilterOption((args->mode >> 1) & 1,
                args->inFilterDegree, sosIn, args->lowpassIn, args->sampleRate, args->epsilon);
    }

    esr = (float) (50. / args->sampleRate);
    filterRet = calloc(filterOutputLength, sizeof(float));

    while (!args->exitFlag) {

        sem_wait(args->full);
        pthread_mutex_lock(&args->mutex);
        memcpy(buf, args->buf, args->bufSize);
        pthread_mutex_unlock(&args->mutex);
        sem_post(args->empty);

        if (!demodMode) {
            convertU8ToReal(buf, args->bufSize, fBuf);
            applyComplexFilter(fBuf, filterRet, args->bufSize, sosLen, sosIn, windIn);
            fwrite(filterRet, sizeof(float), args->bufSize, args->outFile);
        } else {
            processInput(buf, args->bufSize, fBuf);
            if (!args->inFilterDegree) {
                fmDemod(fBuf, args->bufSize, demodRet);
                applyFilter(demodRet, filterRet, outputLen,
                        sosLen, sosOut, windOut);
                fwrite(filterRet, sizeof(float), outputLen, args->outFile);
            } else {
                applyComplexFilter(fBuf, filterRet, args->bufSize, sosLen, sosIn, windIn);
                fmDemod(filterRet, args->bufSize, demodRet);
                applyFilter(demodRet, filterRet + args->bufSize, outputLen, sosLen, sosOut, windOut);
                fwrite(filterRet + args->bufSize, sizeof(float),
                        outputLen, args->outFile);
            }
        }
        memset(filterRet, 0, filterOutputLength * sizeof(float));
    }
    free(windIn);
    free(windOut);
    free(buf);
    free(fBuf);
    free(demodRet);
    free(filterRet);

    return NULL;
}

void allocateBuffer(void **buf, const size_t len) {

    *buf = calloc(len, 1);
}
