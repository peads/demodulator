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

static LREAL samplingRate;
static float esr;

static inline void generateCosSum(const size_t n,
                                  const size_t m,
                                  const LREAL *__restrict__ a,
                                  REAL *__restrict__ wind) {
    size_t k;
    LREAL t, x;
    size_t i, N = n >> 1;

    N = (n & 1) ? N + 1 : N;

    for (i = 0; i < N; ++i) {
        t = 2. * M_PI * (LREAL) i / (N << 1);
        x = 0;
        for (k = 0; k < m; ++k) {
            x += POW(-1, k) * a[k] * COS(k * t);
        }
        wind[n - i - 1] =
            wind[i] = (REAL) x;
#if 0
        fprintf(stderr, "%f ", wind[i]);
#endif
    }
#if 0
    fprintf(stderr, "\n");
#endif
}

static inline void generateHann(const size_t n, REAL *__restrict__ wind) {

    size_t i, N = n >> 1;
    LREAL x;
    N = (n & 1) ? N + 1 : N;
    for (i = 0; i < N; ++i) {
        x = SIN(M_PI * (LREAL) i / (LREAL) (N << 1));
        wind[n - i - 1] =
        wind[i] = (REAL) (x * x);
#if 0
        fprintf(stderr, "%f ", wind[i]);
#endif
    }
#if 0
    fprintf(stderr, "\n");
#endif
}

static inline void generateBH(const size_t n, REAL *__restrict__ wind) {

    static const LREAL a[4] = {0.35875, 0.48829, 0.14128, 0.01168};
    generateCosSum(n, 4, a, wind);
}

static inline void generateBN(const size_t n, REAL *__restrict__ wind) {

    static const LREAL a[4] = {0.3635819, 0.4891775, 0.1365995, 0.0106411};
    generateCosSum(n, 4, a, wind);
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

    switch (mode) {
        case 1:
            wh = COSH(1. / (LREAL) degree * ACOSH(1. / SQRT(POW(10., epsilon) - 1.)));
#ifdef VERBOSE
            fprintf(stderr, PRINT_EP_WC, epsilon * 10., wh * fc);
#endif
            wh = TAN(wh * w);
            transformBilinear(degree, wh, epsilon, 0, warpCheby1, sos);
            break;
        case 2:
            transformBilinear(degree, 1. / SIN(2. * w), TAN(w), 1, warpButterHp, sos);
            break;
        case 0: // fall-through intended
        default:
            transformBilinear(degree, 1. / SIN(2. * w), TAN(w), 0, warpButter, sos);
            break;
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

static inline void convertU8ToReal(
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

        out[i + 1] = ((float) buf[i + 1]) - off[1];
        out[len - i - 1] = ((float) buf[len - i - 1]) - off[1];

        off[0] += (out[i] + out[len - i - 2]) * esr;
        off[1] += (out[i + 1] + out[len - i - 1]) * esr;
    }
}

static inline void highpassDc(
        void *__restrict__ in,
        const size_t len,
        float *__restrict__ out) {

    static REAL *wind = NULL;
    static float sos[2][6];
    static float *buf = NULL;
    if (!wind) {
        wind = calloc(3, sizeof(float));
        generateHann(3, wind);
        processFilterOption(2, 3, sos, 1., samplingRate, 0.);
        buf = calloc(len, sizeof(float));
    }

    convertU8ToReal(in, len, buf);
    applyComplexFilter(buf, out, len, 2, sos, wind);
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
//    float *fBuf = calloc(args->bufSize, sizeof(float));
    float *demodRet = calloc(args->bufSize, sizeof(float));

    const size_t twiceBufSize = args->bufSize << 1;
    size_t filterOutputLength = twiceBufSize;
    size_t filterBytes;

    const size_t outputLen = args->bufSize >> 2;
    const uint8_t demodMode = (args->mode >> 4) & 3;
    const size_t sosLen =
            (args->outFilterDegree & 1) ? (args->outFilterDegree >> 1) + 1
                                        : (args->outFilterDegree >> 1);

    iqCorrection_t processInput = NULL;
    float sosIn[sosLen][6];
    float sosOut[sosLen][6];
    float *windIn = NULL;
    float *windOut = calloc(args->outFilterDegree, sizeof(float));
    windowGenerator_t genWindow;

    samplingRate = args->sampleRate;

    switch ((args->mode >> 2) & 3) {
        case 0:
            processInput = correctIq;
            break;
        case 1:
            processInput = shiftOrigin;
            break;
        case 2:
//            filterOutputLength <<= 1;
            processInput = highpassDc;
            break;
        case 3:
            esr = (float) (50. / args->sampleRate);
            processInput = convertU8ToReal;
            break;
        default:
            break;
    }

    switch ((args->mode >> 6) & 3) {
        case 0:
            genWindow = generateBN;
            break;
        case 2:
            genWindow = generateBH;
            break;
        case 1: // fall-through intended
        case 3:
        default:
            genWindow = generateHann;
            break;
    }

    genWindow(args->outFilterDegree, windOut);

    processFilterOption(args->mode & 1,
            args->outFilterDegree, sosOut, args->lowpassOut, args->sampleRate, args->epsilon);
    if (args->lowpassIn) {
        args->inFilterDegree = args->outFilterDegree; // TODO decouple out and in filter lens
        windIn = calloc(args->inFilterDegree, sizeof(float));
        genWindow(args->inFilterDegree, windIn);
        filterOutputLength <<= 1;
        processFilterOption((args->mode >> 1) & 1,
                args->inFilterDegree, sosIn, args->lowpassIn, args->sampleRate, args->epsilon);
    }

    filterBytes = filterOutputLength * sizeof(float);
    filterRet = calloc(filterOutputLength, sizeof(float));

    while (!args->exitFlag) {

        sem_wait(args->full);
        pthread_mutex_lock(&args->mutex);
        memcpy(buf, args->buf, args->bufSize);
        pthread_mutex_unlock(&args->mutex);
        sem_post(args->empty);

        if (!demodMode) {
            convertU8ToReal(buf, args->bufSize, filterRet);
            applyComplexFilter(filterRet, filterRet + args->bufSize, args->bufSize, sosLen, sosIn, windIn);
            fwrite(filterRet + args->bufSize, sizeof(float), args->bufSize, args->outFile);
        } else {
            processInput(buf, args->bufSize, filterRet);
            if (!args->inFilterDegree) {
                fmDemod(filterRet, args->bufSize, demodRet);
                applyFilter(demodRet, filterRet + args->bufSize, outputLen,
                        sosLen, sosOut, windOut);
                fwrite(filterRet + args->bufSize, sizeof(float), outputLen, args->outFile);
            } else {
                applyComplexFilter(filterRet, filterRet + args->bufSize, args->bufSize, sosLen, sosIn, windIn);
                fmDemod(filterRet + args->bufSize, args->bufSize, demodRet);
                applyFilter(demodRet, filterRet + twiceBufSize, outputLen, sosLen,
                        sosOut, windOut);
                fwrite(filterRet + twiceBufSize, sizeof(float),
                        outputLen, args->outFile);
            }
        }
        memset(filterRet, 0, filterBytes);
    }
    free(windIn);
    free(windOut);
    free(buf);
    free(demodRet);
    free(filterRet);

    return NULL;
}

void allocateBuffer(void **buf, const size_t len) {

    *buf = calloc(len, 1);
}
