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

typedef double (*warpGenerator_t)(size_t, size_t, double, double *);
typedef float (*windowGenerator_t)(size_t, size_t);
static double TAN = NAN; //TODO figure out partially applied fns in C to avoid this

static inline double warpButter(const size_t k,
                                const size_t n,
                                const double theta,
                                double *z) {

    size_t j = (k - 1) << 2;
    const double w = M_PI_2 * (1. / (double) n * (-1. + (double) (k << 1)) + 1.);
    const double a = cos(w);
    const double d = 1. / (a - 1. / sin(2. * theta));
    const double zr = (a - TAN) * d;
    const double zj = sin(w) * d;

    z[j + 2] = z[j] = 1. - zr;
    z[j + 1] = zj;
    z[j + 3] = -zj;

    return zr;
}

static double warpCheby1(const size_t k, const size_t n, const double ep, double *z) {

    size_t j = (k - 1) << 2;
    const double oneOverN = 1. / (double) n;
    const double v = log((1. + pow(10., 0.5 * ep)) / sqrt(pow(10., ep) - 1.)) * oneOverN;
    const double t = M_PI_2 * (oneOverN * (-1. + (double) (k << 1)));

    const double a = cos(t) * cosh(v) * TAN;
    const double b = sin(t) * sinh(v) * TAN;
    const double c = a * a + b * b;
    const double d = 1. / (1. + c + 2. * b);
    double zj = 2. * a * d;
    double zr = 2. * (b + c) * d;

    z[j + 2] = z[j] = 1. - zr;
    z[j + 1] = zj;
    z[j + 3] = -zj;

    return zr;
}

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

static inline void generateCoeffs(const size_t k,
                                  const size_t n,
                                  const double theta,
                                  const warpGenerator_t warp,
                                  double *acc,
                                  double *z) {

    double a, zj;
    double zr = warp(k, n, theta, z);
    zj = z[((k - 1) << 2) + 1]; // 2k - 1

    if (k <= n >> 1) {
        a = zr * zr + zj * zj;
        acc[0] *= a;
        acc[1] *= a;
    } else {
        a = zr * acc[0] - zj * acc[1];
        acc[1] = zr * acc[1] + zj * acc[0];
        acc[0] = a;
    }
#ifdef VERBOSE
    fprintf(stderr, "(%f +/- %f I), ", 1. - zr, zj);
#endif
}

/// Note this simplification will not work for non-bilinear transform transfer functions
static inline void zp2Sos(const size_t n, const double *z, const double *p, const double k, double sos[][6]) {

    size_t i, j;
    size_t npc = n >> 1;
    size_t npr = 0;

    if (n & 1) {
        npr = 1;
    }

    for (j = 0, i = 0; j < npc; i += 4, ++j) {
        sos[j][3] = sos[j][0] = 1.;
        sos[j][1] = -2. * z[i];
        sos[j][2] = z[i] * z[i] + z[i + 1] * z[i + 1];
        sos[j][4] = -2. * p[i];
        sos[j][5] = p[i] * p[i] + p[i + 1] * p[i + 1];
    }

    for (j = npc, i = (n << 1) - npc + 1; j < npc + npr; i += 4, ++j) {
        sos[j][3] = 1.;
        sos[j][2] = sos[j][5] = 0.;
        sos[j][0] = sos[j][1] = k;
        sos[j][4] = -p[i];
    }
}

static inline double transformBilinear(const size_t n,
                                       const double theta,
                                       double sos[][6],
                                       const warpGenerator_t warp) {

    size_t i, k;
    double acc[2] = {1., 0};
    double *p = calloc(((n + 1) << 1), sizeof(double));
    double *z = calloc(((n + 1) << 1), sizeof(double));
    double *t = calloc((n << 1), sizeof(double));
    size_t N = n >> 1;
    N = (n & 1) ? N + 1 : N;
#ifdef VERBOSE
    fprintf(stderr, "\nz: There are n = %zu zeros at z = -1 for (z+1)^n\np: ", n);
#endif
    // Generate roots of bilinear transform
    for (k = 1; k <= N; ++k) {
        generateCoeffs(k, n, theta, warp, acc, p);
    }

    // Store the gain
    acc[0] /= pow(2., (double) n);

    for (i = 0; i < n << 1; i += 2) {
        z[i] = -1.;
        z[i + 1] = 0;
    }

    zp2Sos(n, z, p, acc[0], sos);

#ifdef VERBOSE
    size_t j;
    k = n >> 1;
    k = (n & 1) ? k + 1 : k;
    fprintf(stderr, "\nk: %f\n", acc[0]);
    for (i = 0; i < k; ++i) {
        for (j = 0; j < 6; ++j) {
            fprintf(stderr, "%f ", sos[i][j]);
        }
        fprintf(stderr, "\n");
    }
#endif

    free(p);
    free(t);
    free(z);
    return acc[0];
}

static inline void filterOut(float *__restrict__ x,
                             const size_t len,
                             const size_t sosLen,
                             float *__restrict__ y,
                             const float sos[][6],
                             const windowGenerator_t wind) {

    float *xp, *yp, c[2];
    size_t i, j, m;

    for (i = 0; i < len; ++i) {
        j = i + sosLen;
        xp = &x[j];
        yp = &y[j];
        for (m = 0; m < sosLen; ++m) {

            c[0] = wind(m, sosLen + 1);
            c[1] = wind(m + 1, sosLen + 1);

            yp[m] = 1.f + sos[m][0] * yp[m] + sos[m][1] * yp[m + 1];
            yp[m] -= sos[m][3] + sos[m][4] * c[0] * xp[m] + sos[m][5] * c[1] * xp[m + 1];
        }
    }
}

static inline void filterIn(float *__restrict__ x,
                            const size_t len,
                            const size_t sosLen,
                            float *__restrict__ y,
                            const float sos[][6],
                            const windowGenerator_t wind) {

    float *xp, *yp, c[2] = {};
    size_t i, l, j, m;

    for (i = 0; i < len; i += 2) {

        j = i + sosLen;
        yp = &y[j];
        xp = &x[j];

        for (m = 0; m < sosLen; ++m) {

            l = m << 1;

            c[0] = wind(m, sosLen + 1);
            c[1] = wind(m + 1, sosLen + 1);

            yp[l] = sos[m][0] * yp[l] + sos[m][1] * yp[l + 2];
            yp[l] -= sos[m][3] + sos[m][4] * c[0] * xp[l] + sos[m][5] * c[1] * xp[l + 2];

            yp[l + 1] = sos[m][0] * yp[l + 1] + sos[m][1] * yp[l + 3];
            yp[l + 1] -= sos[m][3] + sos[m][4] * c[0] * xp[l + 1] + sos[m][5] * c[1] * xp[l + 3];
        }
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

    if (mode) {
        TAN = tan(cosh(1. / (double) degree * acosh(1. / sqrt(
                pow(10, epsilon) - 1.))) * w);
#ifdef VERBOSE
        fprintf(stderr, "\nepsilon: %f\nwarp factor: %f", epsilon * 10., TAN);
#endif
        transformBilinear(degree, epsilon, sos, warpCheby1);
    } else {
        TAN = tan(w);
        transformBilinear(degree, w, sos, warpButter);
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
        if (!args->lowpassIn) {
            fmDemod(fBuf, args->bufSize, demodRet);
            filterOut(demodRet, args->bufSize >> 2,
                    sosLen, filterRet, sosOut, generateHannCoefficient);
            fwrite(filterRet, sizeof(float), args->bufSize >> 2, args->outFile);
        } else {
            filterIn(fBuf, args->bufSize, sosLen, filterRet, sosIn, generateHannCoefficient);
            fmDemod(filterRet, args->bufSize, demodRet);
            filterOut(demodRet, args->bufSize >> 2, sosLen,
                    filterRet + args->bufSize, sosOut, generateHannCoefficient);
            fwrite(filterRet + args->bufSize, sizeof(float),
                    args->bufSize >> 2, args->outFile);
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
