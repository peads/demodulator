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

void fmDemod(const uint8_t *__restrict__ buf, const uint32_t len, float *__restrict__ result) {

    uint32_t i;
    float ar, aj, br, bj, zr, zj, y;

    for (i = 0; i < len; i++) {

        ar = (float) (buf[i] + buf[i + 2] - 254);
        aj = (float) (254 - buf[i + 1] - buf[i + 3]);

        br = (float) (buf[i + 4] + buf[i + 6] - 254);
        bj = (float) (buf[i + 5] + buf[i + 7] - 254);

        zr = fmaf(ar, br, -aj * bj);
        zj = fmaf(ar, bj, aj * br);

        union fastSqrtPun pun = {.f = fmaf(zr, zr, zj * zj)};
        y = -pun.f;
        pun.i = -(pun.i >> 1) + 0x5f3759df;
        pun.f *= 0.5f * (3.f + y * pun.f * pun.f);

        zr = 64.f * zj * pun.f * 1.f / fmaf(zr * pun.f, 23.f, 41.f);

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
