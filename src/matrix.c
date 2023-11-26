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

typedef float (*warpGenerator_t)(size_t, size_t, float, float *);
static float TAN = NAN; //TODO figure out partially applied fns in C to avoid this

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

static inline float warpButter(const size_t k,
                               const size_t n,
                               const float theta,
                               float *z) {

    size_t j = (k - 1) << 2;
    const float w = M_PI_2 * (1.f / (float) n * (-1.f + (float) (k << 1)) + 1.f);
    const float a = cosf(w);
    const float d = 1.f / (a - 1.f / sinf(2.f * theta));
    const float zr = (a - TAN) * d;
    const float zj = sinf(w) * d;

    z[j + 2] = z[j] = 1.f - zr;
    z[j + 1] = zj;
    z[j + 3] = -zj;

    return zr;
}

static float warpCheby1(const size_t k, const size_t n, const float ep, float *z) {

    size_t j = (k - 1) << 2;
    const float oneOverN = 1.f / (float) n;
    const float v = logf((1.f + powf(10.f, 0.5f * ep)) / sqrtf(powf(10.f, ep) - 1.f)) * oneOverN;
    const float t = M_PI_2 * (oneOverN * (-1.f + (float) (k << 1)));

    const float a = cosf(t) * coshf(v) * TAN;
    const float b = sinf(t) * sinhf(v) * TAN;
    const float c = a * a + b * b;
    const float d = 1.f / (1.f + c + 2.f * b);
    float zj = 2.f * a * d;
    float zr = 2.f * (b + c) * d;

    z[j + 2] = z[j] = 1.f - zr;
    z[j + 1] = zj;
    z[j + 3] = -zj;

    return zr;
}

static inline void generateCoeffs(const size_t k,
                                  const size_t n,
                                  const float theta,
                                  const warpGenerator_t warp,
                                  float *acc,
                                  float *z) {

    float a, zj;
    float zr = warp(k, n, theta, z);
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
void zp2Sos(const size_t n, float *z, float *p, const float k, float sos[][6]) {

    size_t i, j;
    size_t npc = n >> 1;
    size_t npr = 0;

    if (n & 1) {
        npr = 1;
    }

    for (j = 0, i = 0; j < npc; i += 4, ++j) {
        sos[j][3] = sos[j][0] = 1.f;
        sos[j][1] = -2.f * z[i];
        sos[j][2] = z[i] * z[i] + z[i + 1] * z[i + 1];
        sos[j][4] = -2.f * p[i];
        sos[j][5] = p[i] * p[i] + p[i + 1] * p[i + 1];
    }

    for (j = npc, i = (n << 1) - npc + 1; j < npc + npr; i += 4, ++j) {
        sos[j][3] = sos[j][0] = 1.f;
        sos[j][1] = -z[i];
        sos[j][5] = sos[j][2] = 0.f;
        sos[j][4] = -p[i];
    }
}

static inline float transformBilinear(const size_t n,
                                      const float theta,
                                      float sos[][6],
                                      const warpGenerator_t fn) {

    size_t i, j, k;
    float acc[2] = {1.f, 0};
    float *p = calloc(((n + 1) << 1), sizeof(float));
    float *z = calloc(((n + 1) << 1), sizeof(float));
    float *t = calloc((n << 1), sizeof(float));
    size_t N = n >> 1;
    N = (n & 1) ? N + 1 : N;
#ifdef VERBOSE
    fprintf(stderr, "\nz: There are n = %zu zeros at z = -1 for (z+1)^n\np: ", n);
#endif
    // Generate roots of bilinear transform
    // Perform running sum of coefficients
    // Expand roots into coefficients of monic polynomial
    for (j = 0, k = 1; k <= N; j += 2, ++k) {
        generateCoeffs(k, n, theta, fn, acc, p);
    }

    // Store the gain
    acc[0] /= powf(2.f, (float) n);

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

    zp2Sos(n, z, p, acc[0], sos);

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
    float mva[2] = {};
    for (i = 0; i < len >> 1; i += 2) {
        out[i] = (int8_t) (buf[i] - 127);
        out[i + 1] = (int8_t) (buf[i + 1] - 127);

        out[len - i - 2] = (int8_t) (buf[len - i - 2] - 127);
        out[len - i - 1] = (int8_t) (buf[len - i - 1] - 127);

        mva[0] += (out[i] - out[len - i - 2])/(float)len;
        mva[1] += (out[i+1] - out[len - i - 1])/(float)len ;
    }

    for (i = 0; i < len; i += 2) {
        out[i] -= mva[0];
        out[i+1] -= mva[1];
    }
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
                             const float sos[][6], const float k) {

    float a, b;
    float *xp, *yp;
    size_t i, j, m;

    for (i = 0; i < len; ++i) {
        j = i + filterDegree;
        xp = &x[j];
        yp = &y[j];
        b = a = 0;
        for (m = 0; m < filterDegree; ++m) {
            b += sos[m][0] + sos[m][1] * yp[m] + sos[m][2] * yp[m + 1];
            a += sos[m][3] + sos[m][4] * xp[m] + sos[m][5] * xp[m + 1];
        }
        y[i] = -k*(a - b);
    }
}

static inline void filterIn(float *__restrict__ x,
                            const size_t len,
                            const size_t filterDegree,
                            float *__restrict__ y,
                            const float sos[][6], const float k) {

    float a[2], b[2];
    float *xp, *yp;
    size_t i, j, m;

    for (i = 0; i < len; ++i) {
        j = i + (filterDegree << 1);
        xp = &x[j];
        yp = &y[j];
        b[0] = b[1] = a[0] = a[1] = 0;
        for (m = 0; m < filterDegree; m += 2) {
            j = m << 2;
            a[0] += sos[m][3] + sos[m][4] * xp[j] + sos[m][5] * xp[j + 2];
            b[0] += sos[m][0] + sos[m][1] * yp[j] + sos[m][2] * yp[j + 2];

            a[1] += sos[m][3] + sos[m][4] * xp[j + 1] + sos[m][5] * xp[j + 3];
            b[1] += sos[m][0] + sos[m][1] * yp[j + 1] + sos[m][2] * yp[j + 3];
        }
        y[i] = -k*(a[0] - b[0]);
        y[i + 1] = -k*(a[1] - b[1]);
    }
}

static inline float processFilterOption(uint8_t mode,
                                        size_t degree,
                                        float sos[][6],
                                        float fc,
                                        float fs,
                                        float epsilon) {

    const float w = M_PI * fc / fs;
    float k;

    if (mode) {
        TAN = tanf(coshf(1.f / (float) degree * acoshf(1.f / sqrtf(
                powf(10, epsilon) - 1.f))) * w);
#ifdef VERBOSE
        fprintf(stderr, "\nepsilon: %f\nwarp factor: %f", epsilon * 10.f, TAN);
#endif
        k = transformBilinear(degree, epsilon, sos, warpCheby1);
    } else {
        TAN = tanf(w);
        k = transformBilinear(degree, w, sos, warpButter);
    }

    return k;
}

void *processMatrix(void *ctx) {

    void *buf = calloc(DEFAULT_BUF_SIZE, 1);
    float *fBuf = calloc(DEFAULT_BUF_SIZE, sizeof(float));
    float *demodRet = calloc(DEFAULT_BUF_SIZE, sizeof(float));
    float *filterRet;
    size_t filterOutputLength = DEFAULT_BUF_SIZE;
    consumerArgs *args = ctx;

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

    while (!args->exitFlag) {

        filterRet = calloc(filterOutputLength, sizeof(float));
        sem_wait(args->full);
        pthread_mutex_lock(&args->mutex);
        memcpy(buf, args->buf, DEFAULT_BUF_SIZE);
        pthread_mutex_unlock(&args->mutex);
        sem_post(args->empty);

        shiftOrigin(buf, DEFAULT_BUF_SIZE, fBuf);
        if (!args->lowpassIn) {
            fmDemod(fBuf, DEFAULT_BUF_SIZE, demodRet);
            filterOut(demodRet, DEFAULT_BUF_SIZE >> 2, m, filterRet, sosOut, k);
            fwrite(filterRet, sizeof(float), DEFAULT_BUF_SIZE >> 2, args->outFile);
        } else {
            filterIn(fBuf, DEFAULT_BUF_SIZE, m, filterRet, sosIn, k);
//            balanceIq(filterRet, DEFAULT_BUF_SIZE);
            fmDemod(filterRet, DEFAULT_BUF_SIZE, demodRet);
            filterOut(demodRet, DEFAULT_BUF_SIZE >> 2, m,
                    filterRet + DEFAULT_BUF_SIZE, sosOut, k);
            fwrite(filterRet + DEFAULT_BUF_SIZE, sizeof(float),
                    DEFAULT_BUF_SIZE >> 2, args->outFile);
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
