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
    const float oneOverN = 1.f  / (float) n;
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
        a = zr*zr + zj*zj;
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

static inline void storeCoeffs(size_t n,
                               float *__restrict__ A,
                               float *__restrict__ B,
                               const float *__restrict__ p,
                               const float *__restrict__ acc) {

    size_t k;
    for (k = 0; k < n + 1; ++k) {
        B[k] *= acc[0];
        A[k] = p[k << 1];
    }
}

static inline float transformBilinear(const size_t n,
                                      const float theta,
                                      float *__restrict__ A,
                                      float *__restrict__ B,
                                      const warpGenerator_t fn) {

    size_t i, j, k;
    float acc[2] = {1.f, 0};
    float *p = calloc(((n + 1) << 1), sizeof(float));
    float *z = calloc(((n + 1) << 1), sizeof(float));
    float *t = calloc((n << 1), sizeof(float));
    p[0] = B[0] = B[n] = 1.f;
    size_t N = n >> 1;
    N = (n & 1) ? N + 1 : N;
#ifdef VERBOSE
    fprintf(stderr, "\nz: There are n = %zu zeros at z = -1 for (z+1)^n\np: ", n);
#endif
    // Generate roots of bilinear transform
    // Perform running sum of coefficients
    // Expand roots into coefficients of monic polynomial
    for (j = 0, k = 1; k <= N; j += 2, ++k) {

        B[k] = B[n - k] = B[k - 1] * (float) (n - k + 1) / (float) (k);
        generateCoeffs(k, n, theta, fn, acc, z);
    }

    // TODO remove this once converted to 2nd order section cascade
    for (j = 0; j < n << 1; j += 2) {
        for (i = 0; i <= j; i += 2) {
            t[i] = z[j] * p[i] - z[j + 1] * p[i + 1];
            t[i + 1] = z[j] * p[i + 1] + z[j + 1] * p[i];
        }
        for (i = 0; i < j + 2; i += 2) {
            p[i + 2] -= t[i];
            p[i + 3] -= t[i + 1];
        }
    }

    // Store the output
    acc[0] /= powf(2,(float)n);
    storeCoeffs(n, A, B, p, acc);
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
    for (i = 0; i < len; i += 2) {
        out[i] = (int8_t) (buf[i] - 127);
        out[i + 1] = (int8_t) (buf[i + 1] - 127);
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
                             const size_t filterLen,
                             float *__restrict__ y,
                             const float *__restrict__ A,
                             const float *__restrict__ B) {

    float *xp, *yp, acc;
    size_t i, j, k;

    for (i = 0; i < len; ++i) {
        k = filterLen + i;
        xp = &x[k];
        yp = &y[k];
        acc = 0;
        for (j = 0; j < filterLen; ++j) {
            k = filterLen - j - 1;
            acc += A[k] * xp[j] - B[k] * yp[j];
        }
        y[i] = acc;
    }
}

void filterIn(float *__restrict__ x,
              const size_t len,
              const size_t filterLen,
              float *__restrict__ y,
              const float *__restrict__ A,
              const float *__restrict__ B) {

    float *xp, *yp, acc[2];
    size_t i, j, k, m;

    for (i = 0; i < len; i += 2) {
        k = (filterLen << 1) + i;
        xp = &x[k];
        yp = &y[k];
        acc[0] = acc[1] = 0;
        for (j = 0, m = 0; j < filterLen; ++j, m = j << 1) {
            k = filterLen - j - 1;
            acc[0] += A[k] * xp[m] - B[k] * yp[m];
            acc[1] += A[k] * xp[m + 1] - B[k] * yp[m + 1];
        }
        y[i] = acc[0];
        y[i + 1] = acc[1];
    }
}
static inline float processFilterOption(uint8_t mode, size_t degree, float *A, float *B, float fc, float fs, float epsilon) {

    const float w = M_PI * fc / fs;
    float k;

    if (mode) {
        TAN = tanf(coshf(1.f / (float) degree * acoshf(1.f / sqrtf(
                powf(10, epsilon) - 1.f))) * w);
#ifdef VERBOSE
        fprintf(stderr, "\nepsilon: %f\nwarp factor: %f", epsilon * 10.f, TAN);
#endif
        k = transformBilinear(degree, epsilon, A, B, warpCheby1);
    } else {
        TAN = tanf(w);
        k = transformBilinear(degree, w, A, B, warpButter);
    }

#ifdef VERBOSE
    fprintf(stderr, "\nk: %.5e\nA: ", k);
    for (size_t i = 0; i < degree + 1; ++i) {
        fprintf(stderr, "%f, ", A[i]);
    }
    fprintf(stderr, "\nB: ");
    for (size_t i = 0; i < degree + 1; ++i) {
        fprintf(stderr, "%.5e, ", B[i]);
    }

    fprintf(stderr, "\n");
#endif

    return k;
}

void *processMatrix(void *ctx) {

    float *C = NULL;
    float *D = NULL;
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
    args->epsilon = args->epsilon ? args->epsilon : 0.01f;

    const size_t outFilterLength = args->outFilterDegree + 1;
    const size_t inFilterLength = args->inFilterDegree + 1;
    float *A = calloc(outFilterLength, sizeof(float));
    float *B = calloc(outFilterLength, sizeof(float));

    processFilterOption(args->mode & 1,
            args->outFilterDegree, A, B, args->lowpassOut, args->sampleRate, args->epsilon);

    if (args->lowpassIn) {
        filterOutputLength <<= 1;
        C = calloc(inFilterLength, sizeof(float));
        D = calloc(inFilterLength, sizeof(float));
        processFilterOption((args->mode >> 1) & 1,
                args->inFilterDegree, C, D, args->lowpassIn, args->sampleRate, args->epsilon);
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
            filterOut(demodRet, DEFAULT_BUF_SIZE >> 2, outFilterLength, filterRet, B, A);
            fwrite(filterRet, sizeof(float), DEFAULT_BUF_SIZE >> 2, args->outFile);
        } else {
            filterIn(fBuf, DEFAULT_BUF_SIZE, inFilterLength, filterRet, D, C);
//            balanceIq(filterRet, DEFAULT_BUF_SIZE);
            fmDemod(filterRet, DEFAULT_BUF_SIZE, demodRet);
            filterOut(demodRet, DEFAULT_BUF_SIZE >> 2, outFilterLength,
                    filterRet + DEFAULT_BUF_SIZE, B, A);
            fwrite(filterRet + DEFAULT_BUF_SIZE, sizeof(float),
                    DEFAULT_BUF_SIZE >> 2, args->outFile);
        }
        free(filterRet);
    }
    free(buf);
    free(fBuf);
    free(demodRet);
    free(A);
    free(B);

    if (args->lowpassIn) {
        free(C);
        free(D);
    }

    return NULL;
}

void allocateBuffer(void **buf, const size_t len) {

    *buf = calloc(len, 1);
}
