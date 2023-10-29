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

static inline void fmDemod(const float *__restrict__ in,
                           const size_t len,
                           const float gain,
                           float *__restrict__ out) {
    size_t i;
    float zr, zj;

    for (i = 0; i < len; i += 4) {

        zr = in[i] * in[i + 2] + in[i + 1] * in[i + 3];
        zj = -in[i] * in[i + 3] + in[i + 1] * in[i + 2];
        zr = atan2f(zj, zr);
        out[i >> 1] = zr;
//        zr = 64.f * zj * frcpf(23.f * zr + 41.f * hypotf(zr, zj));
//
//        out[i >> 3] = isnan(zr) ? 0.f : gain ? zr * gain : zr;
    }
}
void filterLowpass(float *__restrict__ buf, size_t len) {
    static const float BW_CONSTS[16][2] = {
            {-0.0980171f, 0.995185f},
            {-0.290285f, 0.95694f},
            {-0.471397f,  0.881921f},
            {-0.634393f,  0.77301f},
            {-0.77301f,   0.634393f},
            {-0.881921f,  0.471397f},
            {-0.95694f,   0.290285f},
            {-0.995185f,  0.0980171f},
            {-0.995185f,  -0.0980171f},
            {-0.95694f,   -0.290285f},
            {-0.881921f,  -0.471397f},
            {-0.77301f,   -0.634393f},
            {-0.634393f,-0.77301f},
            {-0.471397f,-0.881921f},
            {-0.290285f,-0.95694f},
            {-0.0980171f,-0.995185f}};
    static const float Wc = 2500.f;
    size_t i,j;
    float accR, accJ, currR, currJ;//, normSquared;
    const float *constPtr;

    for (i = 0; i < len; i+=2) {
        accR = 1.f;
        accJ = 1.f;
        currR = buf[i] / Wc;
        currJ = buf[i + 1] / Wc;
        for (j = 0; j < 16; ++j) {
            constPtr = BW_CONSTS[j];
            accR *= currR - constPtr[0];
            accJ *= currJ - constPtr[1];
        }
//        normSquared = accR * accR + accJ * accJ;
        buf[i] = buf[i]/accR;//accR / normSquared * buf[i];
        buf[i + 1] = buf[i+1]/accJ;//accJ / normSquared * buf[i + 1];
    }
}

void filterHighpass(float *__restrict__ buf, size_t len) {
    static const float BW_CONSTS[16][2] = {
           {-0.0980171f, 0.995185f},
           {-0.290285f, 0.95694f},
           {-0.471397f,  0.881921f},
           {-0.634393f,  0.77301f},
           {-0.77301f,   0.634393f},
           {-0.881921f,  0.471397f},
           {-0.95694f,   0.290285f},
           {-0.995185f,  0.0980171f},
           {-0.995185f,  -0.0980171f},
           {-0.95694f,   -0.290285f},
           {-0.881921f,  -0.471397f},
           {-0.77301f,   -0.634393f},
           {-0.634393f,-0.77301f},
           {-0.471397f,-0.881921f},
           {-0.290285f,-0.95694f},
           {-0.0980171f,-0.995185f}};
    static const float Wc = 1.f;
    size_t i,j;
    float accR, accJ, currR, currJ;//, normSquared;
    const float *constPtr;

    for (i = 0; i < len; i+=2) {
        accR = 1.f;
        accJ = 1.f;
        currR = Wc / buf[i];
        currJ = Wc / buf[i + 1];
        for (j = 0; j < 16; ++j) {
            constPtr = BW_CONSTS[j];
            accR *= currR - constPtr[0];
            accJ *= currJ - constPtr[1];
        }
//        normSquared = accR * accR + accJ * accJ;
        buf[i] = buf[i]/accR;//accR / normSquared * buf[i];
        buf[i + 1] = buf[i+1]/accJ;//accJ / normSquared * buf[i + 1];
    }
}

//void crudeLowpass(void *__restrict__ in, size_t len, float *__restrict__ out) {
//
//    size_t i, j;
//    int8_t *buf = in;
//    for (i = 0; i < len; i+=4) {
//        j = i>>1;
//        out[j] = (float) (buf[i] + buf[i + 2] + buf[i + 4] + buf[i + 6] - 508);
//        out[j+1] = (float) (buf[i + 1] + buf[i + 3] + buf[i + 5] + buf[i + 7] - 508);
//    }
//}

void shiftOrigin(void *in, long len, float *out) {

    size_t i;
    int8_t *buf = in;
    for (i = 0; i < len; i+=2) {
        out[i] = (int8_t)(buf[i] - 127);
        out[i+1] = (int8_t)(buf[i+1] - 127);
    }
}

void *processMatrix(void *ctx) {

    consumerArgs *args = ctx;
    void *buf = calloc(DEFAULT_BUF_SIZE, 1);
    float *fBuf = calloc(DEFAULT_BUF_SIZE, sizeof(float));
    float *result = calloc(DEFAULT_BUF_SIZE>>1, sizeof(float));

    while (!args->exitFlag) {

        sem_wait(args->full);
        pthread_mutex_lock(&args->mutex);
        memcpy(buf, args->buf, DEFAULT_BUF_SIZE);
        pthread_mutex_unlock(&args->mutex);
        sem_post(args->empty);

        shiftOrigin(buf, DEFAULT_BUF_SIZE, fBuf);
//        crudeLowpass(buf, DEFAULT_BUF_SIZE, fBuf);
        filterLowpass(fBuf, DEFAULT_BUF_SIZE);
        filterHighpass(fBuf, DEFAULT_BUF_SIZE);
        fmDemod(fBuf, DEFAULT_BUF_SIZE, args->gain, result);
        fwrite(result, sizeof(float), DEFAULT_BUF_SIZE >> 1, args->outFile);
    }
    free(buf);
    free(fBuf);
    free(result);

    return NULL;
}

void allocateBuffer(void **buf, const size_t len) {

    *buf = calloc(len, 1);
}
