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
static REAL esr;

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
    }
}

static inline void generateRect(const size_t n, REAL *__restrict__ wind) {

    size_t i, N = n >> 1;
    N = (n & 1) ? N + 1 : N;
    for (i = 0; i < N; ++i) {
        wind[n - i - 1] =
        wind[i] = (REAL) 1.;
    }
}

static inline void generateHann(const size_t n, REAL *__restrict__ wind) {

    static const LREAL a[2] = {0.5, 0.5};
    generateCosSum(n, 2, a, wind);
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
                                       REAL sosf[][6],
                                       LREAL fc,
                                       LREAL fs,
                                       LREAL epsilon) {

    size_t N = degree >> 1;
    N = (degree & 1) ? N + 1 : N;
    const LREAL w = 2. * M_PI * M_PI * fc / fs;
    size_t i, j;
    LREAL sos[N][6];
    const LREAL wh = COSH(1. / (LREAL) degree * ACOSH(1. / SQRT(POW(10., epsilon) - 1.)));

#ifdef VERBOSE
    fprintf(stderr, "\ndegree: %zu", degree);
    if (3 == mode || 1 == mode) {
        fprintf(stderr, PRINT_EP_WC, epsilon * 10., wh * fc);
    }
#endif

    switch (mode) {
        case 1:
            transformBilinear(degree, TAN(w * wh), epsilon, 0, warpCheby1, sos);
            break;
        case 2:
            transformBilinear(degree, 1. / SIN(2. * w), TAN(w), 1, warpButterHp, sos);
            break;
        case 3:
            transformBilinear(degree, TAN(w * wh), epsilon, 1, warpCheby1Hp, sos);
            break;
        default:
            transformBilinear(degree, 1. / SIN(2. * w), TAN(w), 0, warpButter, sos);
            break;
    }

    for (i = 0; i < N; ++i) {
        for (j = 0; j < 6; ++j) {
            sosf[i][j] = (REAL) sos[i][j];
        }
    }
}

static inline void shiftOrigin(
        void *__restrict__ in,
        const size_t len,
        REAL *__restrict__ out) {

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
        REAL *__restrict__ out) {

    const size_t N = len >> 1;
    size_t i;
    uint8_t *buf = in;

    for (i = 0; i < N; i += 2) {
        out[i] = (REAL) buf[i];
        out[i + 1] = (REAL) buf[i + 1];

        out[len - i - 2] = (REAL) buf[len - i - 2];
        out[len - i - 1] = (REAL) buf[len - i - 1];
    }
}

static inline void correctIq(
        void *__restrict__ in,
        const size_t len,
        REAL *__restrict__ out) {

    static REAL off[2] = {};
    const size_t N = len >> 1;
    size_t i;
    uint8_t *buf = in;

    for (i = 0; i < N; i += 2) {
        out[i] = ((REAL) buf[i]) - off[0];
        out[len - i - 2] = ((REAL) buf[len - i - 2]) - off[0];

        out[i + 1] = ((REAL) buf[i + 1]) - off[1];
        out[len - i - 1] = ((REAL) buf[len - i - 1]) - off[1];

        off[0] += (out[i] + out[len - i - 2]) * esr;
        off[1] += (out[i + 1] + out[len - i - 1]) * esr;
    }
}

static inline void highpassDc(
        void *__restrict__ in,
        const size_t len,
        REAL *__restrict__ out) {

    static const size_t degree = 3;
    static const size_t sosLen = 2;
    // TODO parameterize degree?
//            (degree & 1) ? (degree >> 1) + 1 : degree >> 1;
    static REAL *wind = NULL;
    static REAL sos[2][6];
    static REAL *buf = NULL;

    if (!wind) {
        wind = calloc(degree, sizeof(REAL));
        generateRect(degree, wind);
        processFilterOption(2, degree, sos, 1., samplingRate, 0.);
        buf = calloc(len, sizeof(REAL));
    }


    convertU8ToReal(in, len, buf);
    applyComplexFilter(buf, out, len, sosLen, sos, wind);
}

static inline void fmDemod(const REAL *__restrict__ in,
                           const size_t len,
                           REAL *__restrict__ out) {

    size_t i;
    REAL zr, zj;

    for (i = 0; i < len; i += 4) {

        // ac-b(-d)=ac+bd
        // a(-d)+bc=-ad+bc
        zr = in[i] * in[i + 2] + in[i + 1] * in[i + 3];
        zj = -in[i] * in[i + 3] + in[i + 1] * in[i + 2];

        zr = (REAL) (64. * zj * 1. / (23. * zr + 41. * HYPOTF(zr, zj)));
        out[i >> 2] = isnan(zr) ? (REAL) 0. : zr;
    }
}

void *processMatrix(void *ctx) {

    consumerArgs *args = ctx;
    void *buf = calloc(args->bufSize, 1);

    REAL *filterRet;
    REAL *demodRet = calloc(args->bufSize, sizeof(REAL));

    const size_t twiceBufSize = args->bufSize << 1;
    size_t filterOutputLength = twiceBufSize;
    size_t filterBytes;

    args->inFilterDegree = !args->lowpassIn && !args->inFilterDegree ? args->inFilterDegree : args->outFilterDegree;
    const size_t outputLen = args->bufSize >> 2;
    const uint8_t demodMode = (args->mode >> 4) & 3;
    const size_t sosLenOut =
            (args->outFilterDegree & 1) ? (args->outFilterDegree >> 1) + 1
                                        : (args->outFilterDegree >> 1);
    const size_t sosLenIn =
            (args->inFilterDegree & 1) ? (args->inFilterDegree >> 1) + 1
                                        : (args->inFilterDegree >> 1);

    iqCorrection_t processInput = NULL;
    REAL sosIn[sosLenIn][6];
    REAL sosOut[sosLenOut][6];
    REAL *windIn = NULL;
    REAL *windOut = calloc(args->outFilterDegree, sizeof(REAL));
    windowGenerator_t genWindow;

    samplingRate = args->sampleRate;

    switch ((args->mode >> 2) & 3) {
        case 1:
            processInput = correctIq;
            break;
        case 2:
            processInput = highpassDc;
            break;
        case 3:
            esr = (REAL) (50. / args->sampleRate);
            processInput = convertU8ToReal;
            break;
        default:
            processInput = shiftOrigin;
            break;
    }

    switch ((args->mode >> 6) & 3) {
        case 1:
            genWindow = generateBN;
            break;
        case 2:
            genWindow = generateBH;
            break;
        case 3:
            genWindow = generateRect;
            break;
        default:
            genWindow = generateHann;
            break;
    }

    genWindow(args->outFilterDegree, windOut);

    processFilterOption(args->mode & 1,
            args->outFilterDegree, sosOut, args->lowpassOut, args->sampleRate, args->epsilon);

    if (args->lowpassIn) {
        windIn = calloc(args->inFilterDegree, sizeof(REAL));
        genWindow(args->inFilterDegree, windIn);
        filterOutputLength <<= 1;
        processFilterOption((args->mode >> 1) & 1,
                args->inFilterDegree, sosIn, args->lowpassIn, args->sampleRate, args->epsilon);
    }

    filterBytes = filterOutputLength * sizeof(REAL);
    filterRet = calloc(filterOutputLength, sizeof(REAL));

    while (!args->exitFlag) {

        sem_wait(args->full);
        pthread_mutex_lock(&args->mutex);
        memcpy(buf, args->buf, args->bufSize);
        pthread_mutex_unlock(&args->mutex);
        sem_post(args->empty);

        if (!demodMode) {
            convertU8ToReal(buf, args->bufSize, filterRet);
            applyComplexFilter(filterRet, filterRet + args->bufSize, args->bufSize, sosLenIn, sosIn, windIn);
            fwrite(filterRet + args->bufSize, sizeof(REAL), args->bufSize, args->outFile);
        } else {
            processInput(buf, args->bufSize, filterRet);
            if (!args->inFilterDegree) {
                fmDemod(filterRet, args->bufSize, demodRet);
                applyFilter(demodRet, filterRet + args->bufSize, outputLen,
                        sosLenOut, sosOut, windOut);
                fwrite(filterRet + args->bufSize, sizeof(REAL), outputLen, args->outFile);
            } else {
                applyComplexFilter(filterRet, filterRet + args->bufSize, args->bufSize, sosLenIn, sosIn, windIn);
                fmDemod(filterRet + args->bufSize, args->bufSize, demodRet);
                applyFilter(demodRet, filterRet + twiceBufSize, outputLen, sosLenOut,
                        sosOut, windOut);
                fwrite(filterRet + twiceBufSize, sizeof(REAL),
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
