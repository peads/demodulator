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
#include <math.h>
#include <stdlib.h>
#include "definitions.h"
#include "matrix.h"

//float quickRsqrt(float f) {
//
//    union {
//        float f;
//        uint32_t i;
//    } punned = {.f = f};
//
//    punned.i = -(punned.i >> 1) + 0x5f3759df;
//    punned.f *= -0.5f * (-3.f + punned.f * punned.f * f);
//    return punned.f;
//}

float quickRsqrtf(float x) {

    //float xhalf = 0.5f * x;
    float y = -x;
    uint32_t i = *(uint32_t *) &x;
    i = -(i >> 1) + 0x5f3759df;
    x = *(float *) &i;
    x *= 0.5f * (3.f + y * x * x);
    return x;
}

void fmDemod(const uint8_t *__restrict__ buf, const uint32_t len, float *__restrict__ result) {

    uint32_t i;
    float ar, aj, br, bj, zr, zj, lenR;

    for (i = 0; i < len; i++) {

        ar = (float) (buf[i] + buf[i + 2] - 254);
        aj = (float) (254 - buf[i + 1] - buf[i + 3]);

        br = (float) (buf[i + 4] + buf[i + 6] - 254);
        bj = (float) (buf[i + 5] + buf[i + 7] - 254);

        zr = fmaf(ar, br, -aj * bj);
        zj = fmaf(ar, bj, aj * br);

        lenR = quickRsqrtf(fmaf(zr, zr, zj * zj));
//        lenR = 1.f / sqrtf(fmaf(zr, zr, zj * zj));

        zr = 64.f * zj * lenR * 1.f / fmaf(zr * lenR, 23.f, 41.f);

        result[i >> 2] = isnan(zr) ? 0.f : zr;
    }
}

int processMatrix(float squelch, FILE *inFile, struct chars *chars, void *outFile) {

    uint8_t *buf = calloc(DEFAULT_BUF_SIZE, INPUT_ELEMENT_BYTES);
    int exitFlag = 0;
    size_t readBytes;
    float result[QTR_BUF_SIZE];

    while (!exitFlag) {

        readBytes = fread(buf + 2, INPUT_ELEMENT_BYTES, DEFAULT_BUF_SIZE - 2, inFile);

        if ((exitFlag = ferror(inFile))) {
            perror(NULL);
            break;
        } else if (feof(inFile)) {
            exitFlag = EOF;
        }

        fmDemod(buf, readBytes, result);

        fwrite(result, OUTPUT_ELEMENT_BYTES, QTR_BUF_SIZE, outFile);
    }
    return exitFlag;
}
