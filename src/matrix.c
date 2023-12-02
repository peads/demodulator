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

typedef double (*warpGenerator_t)(size_t, size_t, const double, double *);
typedef float (*windowGenerator_t)(size_t, size_t);
static double TAN = NAN;

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
//static float *windowIn = NULL;
    static float *windowOut = NULL;
    if (!windowOut) {
        size_t i, N = n >> 1;
        N = (n & 1) ? N + 1 : N;
        windowOut = calloc(n, sizeof(float));
        double x;
        for (i = 0; i < N; ++i) {
            x = sin(M_PI * (double) (i + 1) / (double) (n + 1));
            x *= x;
            windowOut[n - i - 1] = windowOut[i] = (float) x;
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
    fprintf(stderr, "(%f +/- %f I), ", 1.f - zr, zj);
#endif
}

/// Note this simplification will not work for non-bilinear transform transfer functions
void zp2Sos(const size_t n, const double *z, const double *p, double sos[][6]) {

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
        sos[j][3] = sos[j][0] = 1.;
        sos[j][1] = -z[i];
        sos[j][5] = sos[j][2] = 0.;
        sos[j][4] = -p[i];
    }
}

static inline double transformBilinear(const size_t n,
                                      const double theta,
                                      double sos[][6],
                                      const warpGenerator_t warp) {

    size_t i, j, k;
    double acc[2] = {1.f, 0};
    double *p = calloc(((n + 1) << 1), sizeof(double));
    double *z = calloc(((n + 1) << 1), sizeof(double));
    double *t = calloc((n << 1), sizeof(double));
    size_t N = n >> 1;
    N = (n & 1) ? N + 1 : N;
#ifdef VERBOSE
    fprintf(stderr, "\nz: There are n = %zu zeros at z = -1 for (z+1)^n\np: ", n);
#endif
    // Generate roots of bilinear transform
    // Perform running sum of coefficients
    // Expand roots into coefficients of monic polynomial
    for (j = 0, k = 1; k <= N; j += 2, ++k) {
        generateCoeffs(k, n, theta, warp, acc, p);
    }

    // Store the gain
    acc[0] /= pow(2., (double) n);

    for (i = 0; i < n << 1; i += 2) {
        z[i] = -1.f;
        z[i + 1] = 0;
    }

    k = n >> 1;
    k = (n & 1) ? k + 1 : k;
    for (i = 0; i < k; ++i) {
        for (j = 0; j < 6; ++j) {
            sos[i][j] = 0;
        }
    }

    zp2Sos(n, z, p, sos);

#ifdef VERBOSE
    fprintf(stderr, "\n");
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

static inline void shiftOrigin(
        void *__restrict__ in,
        const size_t len,
        float *__restrict__ out) {

    size_t i;
    int8_t *buf = in;
//    float mva[2] = {};
    for (i = 0; i < len >> 1; i += 2) {
        out[i] = (int8_t) (buf[i] - 127);
        out[i + 1] = (int8_t) (buf[i + 1] - 127);

        out[len - i - 2] = (int8_t) (buf[len - i - 2] - 127);
        out[len - i - 1] = (int8_t) (buf[len - i - 1] - 127);

//        mva[0] += (out[i] - out[len - i - 2]) / (float) len;
//        mva[1] += (out[i + 1] - out[len - i - 1]) / (float) len;
    }

//    for (i = 0; i < len; i += 2) {
//        out[i] -= mva[0];
//        out[i + 1] -= mva[1];
//    }
}

inline void balanceIq(float *__restrict__ buf, const size_t len) {

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
                             const size_t filterDegree,
                             float *__restrict__ y,
                             const float sos[][6],
                             const float k,
                             const windowGenerator_t wind) {

    float a, b;
    float *xp, *yp, c[2];
    size_t i, j, m;

    for (i = 0; i < len; ++i) {
        j = i + filterDegree;
        xp = &x[j];
        yp = &y[j];
        b = a = 0;
        for (m = 0; m < filterDegree; ++m) {

            c[0] = wind(m, filterDegree);
            c[1] = wind(m + 1, filterDegree);

            b += sos[m][0] + sos[m][1] * yp[m] + sos[m][2] * yp[m + 1];
            a += sos[m][3] + sos[m][4] * c[0] * xp[m] + sos[m][5] * c[1] * xp[m + 1];
        }
        y[i] = -k * (a - b);
    }
}

static inline void filterIn(float *__restrict__ x,
                            const size_t len,
                            const size_t filterDegree,
                            float *__restrict__ y,
                            const float sos[][6],
                            const float k,
                            const windowGenerator_t wind) {

    float a[2], b[2];
    float *xp, *yp, c[2] = {};
    size_t i, l, j, m;

    for (i = 0; i < len; i += 2) {

        j = i + (filterDegree << 1);
        yp = &y[i];
        xp = &x[j];
        b[0] = b[1] = a[0] = a[1] = 0;

        for (m = 0; m < filterDegree; ++m) {

            l = m << 1;

            c[0] = wind(m, filterDegree);
            c[1] = wind(m + 1, filterDegree);

            a[0] += sos[m][3] + sos[m][4] * c[0] * xp[l] + sos[m][5] * c[1] * xp[l + 2];
            b[0] += sos[m][0] + sos[m][1] * yp[l] + sos[m][2] * yp[l + 2];

            a[1] += sos[m][3] + sos[m][4] * c[0] * xp[l + 1] + sos[m][5] * c[1] * xp[l + 3];
            b[1] += sos[m][0] + sos[m][1] * yp[l + 1] + sos[m][2] * yp[l + 3];
        }

        y[i] = -k * (a[0] - b[0]);
        y[i + 1] = -k * (a[1] - b[1]);
    }
}

static inline double processFilterOption(uint8_t mode,
                                        size_t degree,
                                        size_t m,
                                        float sosf[][6],
                                        double fcp,
                                        double fs,
                                        double epsilon) {

    const double w = M_PI * fcp / fs;
    double sos[m][6];
    double wh;
    double k;
    size_t i, j;

    if (mode) {
        wh = cosh(1. / (double) degree * acosh(1. / sqrt(
                pow(10.f, epsilon) - 1.)));
        TAN = tan(w * wh);
        k = transformBilinear(degree, epsilon, sos, warpCheby1);
#ifdef VERBOSE
        fprintf(stderr, "\nepsilon: %f\nwarp factor: %f\nomegaH: %f\nk: %e", epsilon * 10., TAN, fcp * wh, k);
#endif
    } else {
        TAN = tan(w);
        k = transformBilinear(degree, w, sos, warpButter);
    }

#ifdef VERBOSE
    fprintf(stderr, "\n");
#endif
    for (i = 0; i < m; ++i) {
        for (j = 0; j < 6; ++j) {
            sosf[i][j] = (float) sos[i][j];

#ifdef VERBOSE
            fprintf(stderr, "%f ", sosf[i][j]);
#endif
        }
#ifdef VERBOSE
        fprintf(stderr, "\n");
#endif
    }

    return k;
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

    size_t m = args->outFilterDegree >> 1;
    m = (args->outFilterDegree & 1) ? m + 1 : m;
    float k = 1.f;
    float sosIn[m][6];
    float sosOut[m][6];
    float sosHp[m][6];
    uint8_t outFilterType = args->mode & 1;

    if (!args->lowpassIn) {
        processFilterOption(outFilterType,
                args->outFilterDegree, m, sosOut, args->lowpassOut, args->sampleRate, args->epsilon);
    } else {
        processFilterOption(outFilterType,
                args->outFilterDegree, m, sosOut, args->lowpassOut, args->sampleRate, args->epsilon);
        filterOutputLength <<= 1;
        processFilterOption((args->mode >> 1) & 1,
                args->outFilterDegree, m, sosIn, args->lowpassIn, args->sampleRate, args->epsilon);
    }

    processFilterOption(0, args->outFilterDegree, m, sosHp, 1.f, args->sampleRate, 0);
    while (!args->exitFlag) {

        filterRet = calloc(filterOutputLength, sizeof(float));
        sem_wait(args->full);
        pthread_mutex_lock(&args->mutex);
        memcpy(buf, args->buf, args->bufSize);
        pthread_mutex_unlock(&args->mutex);
        sem_post(args->empty);

        shiftOrigin(buf, args->bufSize, fBuf);
        if (!args->lowpassIn) {
            fmDemod(fBuf, args->bufSize, demodRet);
            filterOut(demodRet, args->bufSize >> 2, m, filterRet, sosOut, k,
                    generateHannCoefficient);
            fwrite(filterRet, sizeof(float), args->bufSize >> 2, args->outFile);
        } else {
            filterIn(fBuf, args->bufSize, m, filterRet, sosIn, k, generateHannCoefficient);
//            balanceIq(filterRet, args->bufSize);
            fmDemod(filterRet, args->bufSize, demodRet);
            filterOut(demodRet, args->bufSize >> 2, m,
                    filterRet + args->bufSize, sosOut, k, generateHannCoefficient);
            fwrite(filterRet + args->bufSize, sizeof(float),
                    args->bufSize >> 2, args->outFile);
        }
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
