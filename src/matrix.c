/*
 * This file is part of the demodulator distribution
 * (https://github.com/peads/demodulator).
 * with code originally part of the misc_snippets distribution
 * (https://github.com/peads/misc_snippets).
 * Copyright (c) 2023-2024 Patrick Eads.
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

static inline void processFilterOption(uint8_t mode,
                                       size_t degree,
                                       REAL sosf[][6],
                                       LREAL fc,
                                       LREAL fs,
                                       LREAL epsilon) {

    size_t N = degree >> 1;
    N = (degree & 1) ? N + 1 : N;
    const LREAL w = M_PI * fc / fs;
    size_t i, j;
    LREAL sos[N][6];
    const LREAL wh = COSH(1. / (LREAL) degree * ACOSH(1. / SQRT(POW(10., epsilon) - 1.)));

#ifdef VERBOSE
    fprintf(stderr,
            "\nCutoff frequency: %.1Lf\nSampling frequency: %.1Lf\nRatio of cutoff frequency to sampling frequency: %.1Lf\ndegree: %zu",
            fc, fs, w, degree);
    if (3 == mode || 1 == mode) {
        fprintf(stderr, PRINT_EP_WC, epsilon * 10., wh * fc);
    }
#endif

    switch (mode) {
        case 1:
#ifdef VERBOSE
            fprintf(stderr, "\nLowpass Chebychev Type 1 selected");
#endif
            transformBilinear(degree, TAN(w * wh), epsilon, 0, warpCheby1, sos);
            break;
        case 2:
#ifdef VERBOSE
            fprintf(stderr, "\nHighpass Butterworth selected");
#endif
            transformBilinear(degree, 1. / SIN(2. * w), TAN(w), 1, warpButterHp, sos);
            break;
        case 3:
#ifdef VERBOSE
            fprintf(stderr, "\nHighpass Chebychev Type 1 selected");
#endif
            transformBilinear(degree, TAN(w * wh), epsilon, 1, warpCheby1Hp, sos);
            break;
        default:
#ifdef VERBOSE
            fprintf(stderr, "\nLowpass Butterworth selected");
#endif
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

static inline void normalizeInput(
        void *__restrict__ in,
        const size_t len,
        REAL *__restrict__ out) {

    // could loop thru entire data to find true max/min, but this is good enough
    const REAL denom = 2. / 255.;
    const size_t N = len >> 1;
    size_t i;
    uint8_t *buf = in;

    for (i = 0; i < N; i += 2) {
        out[i] = (REAL) buf[i] * denom - 1.;
        out[i + 1] = (REAL) buf[i + 1] * denom - 1.;

        out[len - i - 2] = (REAL) buf[len - i - 2] * denom - 1.;
        out[len - i - 1] = (REAL) buf[len - i - 1] * denom - 1.;
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

    static REAL sos[2][6];
    static REAL *buf = NULL;

    if (!buf) {
        processFilterOption(2, 3, sos, 1., samplingRate, 0.);
        buf = calloc(len, sizeof(REAL));
    }

    shiftOrigin(in, len, buf);
    applyComplexFilter(buf, out, len, 2, sos);
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

        zr = (REAL) ATAN2F(zj, zr);
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

    args->inFilterDegree = args->lowpassIn && !args->inFilterDegree
                           ? args->outFilterDegree
                           : args->inFilterDegree;
    const size_t outputLen = args->bufSize >> 2;
    const uint8_t demodMode = (args->mode >> 4) & 3;
    const size_t sosLenOut = (args->outFilterDegree & 1)
                             ? (args->outFilterDegree >> 1) + 1
                             : (args->outFilterDegree >> 1);
    const size_t sosLenIn = (args->inFilterDegree & 1)
                            ? (args->inFilterDegree >> 1) + 1
                            : (args->inFilterDegree >> 1);

    iqCorrection_t processInput = NULL;
    REAL sosIn[sosLenIn][6];
    REAL sosOut[sosLenOut][6];

    samplingRate = args->sampleRate;

    switch ((args->mode >> 2) & 3) {
        case 1:
            esr = (REAL) (50. / args->sampleRate);
            processInput = correctIq;
            break;
        case 2:
            processInput = highpassDc;
            break;
        case 3:
            processInput = normalizeInput;
            break;
        default:
            processInput = shiftOrigin;
            break;
    }

    processFilterOption(args->mode & 1,
            args->outFilterDegree, sosOut, args->lowpassOut, args->sampleRate, args->epsilon);

    if (args->lowpassIn) {
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
            normalizeInput(buf, args->bufSize, filterRet);
            applyComplexFilter(filterRet,
                    filterRet + args->bufSize,
                    args->bufSize,
                    sosLenIn,
                    sosIn);
            fwrite(filterRet + args->bufSize, sizeof(REAL), args->bufSize, args->outFile);
        } else {
            processInput(buf, args->bufSize, filterRet);
            if (!args->inFilterDegree) {
                fmDemod(filterRet, args->bufSize, demodRet);
                applyFilter(demodRet, filterRet + args->bufSize, outputLen,
                        sosLenOut, sosOut);
                fwrite(filterRet + args->bufSize, sizeof(REAL), outputLen, args->outFile);
            } else {
                applyComplexFilter(filterRet,
                        filterRet + args->bufSize,
                        args->bufSize,
                        sosLenIn,
                        sosIn);
                fmDemod(filterRet + args->bufSize, args->bufSize, demodRet);
                applyFilter(demodRet, filterRet + twiceBufSize, outputLen, sosLenOut,
                        sosOut);
                fwrite(filterRet + twiceBufSize, sizeof(REAL),
                        outputLen, args->outFile);
            }
        }
        memset(filterRet, 0, filterBytes);
    }

    free(buf);
    free(demodRet);
    free(filterRet);

    return NULL;
}

void allocateBuffer(void **buf, const size_t len) {

    *buf = calloc(len, 1);
}
