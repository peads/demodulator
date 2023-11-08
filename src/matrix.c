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

static const float coeffB[] = {1.f, 3.f, 3.f, 1.f};

static inline float butter(float fs, float fc, float *__restrict__ coeff) {

    const float ANG = M_PI * fc / fs;
    const float COS = cosf(ANG);
    const float SIN = sinf(ANG);
    const float K = 2.f * SIN * SIN * SIN / ((COS + SIN) * (2.f + sinf(2.f * ANG)));

    size_t i;

    for (i = 0; i < 4; ++i) {
        coeff[i] = coeffB[i] * K;
    }

    return K;
    // TODO QR decomp for n-degree poly?
//    for(k = 0; k < 2; ++k) {
//        coeff[k] = expf(
//                lgammaf((float)NP1)
//                - lgammaf((float)(k+1))
//                - lgammaf((float)(NP1-k)));
//        sumB += coeff[k];
//    }
//    coeff[n] = 1;
//    return sumB + 1;
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
//void blockDc(float fs, float *x, size_t len) {
//    static float K = 0.f;
//    K = !K ? 0.75f * tanf(M_PI_2 / fs) : K;
//    size_t i;
//    for (i = 0; i < len-2; ++i) {
//        x[i] = K * (x[i+2] + x[i+1]) - x[i];
//    }
//}

//void balanceIq(float *__restrict__ buf, size_t len) {
//
//    static const float alpha = 0.99212598425f;
//    static const float beta = 0.00787401574f;
//
//    size_t i;
//    for (i = 0; i < len; i += 2) {
//        buf[i] *= alpha;
//        buf[i + 1] += beta * buf[i];
//    }
//}

static inline void filterOut(float *__restrict__ x,
               size_t len,
               size_t filterLen,
               float *__restrict__ y,
               const float *__restrict__ coeffA) {

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

    static const float fs = 125000.f;
    consumerArgs *args = ctx;
    void *buf = calloc(DEFAULT_BUF_SIZE, 1);
    float *fBuf = calloc(DEFAULT_BUF_SIZE, sizeof(float));
    float *demodRet = calloc(DEFAULT_BUF_SIZE, sizeof(float));
    float coeffLow[4];
//    float coeffHigh[4] = {1,-3,3,-1};
    butter(fs, 15000.f, coeffLow);
    while (!args->exitFlag) {

        float *filterRet = calloc(DEFAULT_BUF_SIZE, sizeof(float));
        sem_wait(args->full);
        pthread_mutex_lock(&args->mutex);
        memcpy(buf, args->buf, DEFAULT_BUF_SIZE);
        pthread_mutex_unlock(&args->mutex);
        sem_post(args->empty);

        shiftOrigin(buf, DEFAULT_BUF_SIZE, fBuf);
        fmDemod(fBuf, DEFAULT_BUF_SIZE, demodRet);
        filterOut(demodRet, DEFAULT_BUF_SIZE >> 2, 4, filterRet, coeffLow);
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
