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

void noconversion(const void *__restrict__ in, const uint32_t index,
                  float *__restrict__ out)  {

    const float *buf = (float *) in;

    out[0] = (buf[index] + buf[index + 2]);
    out[1] = (buf[index + 1] + buf[index + 3]);
    out[2] = (buf[index + 4] + buf[index + 6]);
    out[3] = -(buf[index + 5] + buf[index + 7]);
}

void convertInt16ToFloat(const void *__restrict__ in, const uint32_t index,
                         float *__restrict__ out) {

    const int16_t *buf = (int16_t *) in;

    out[0] = (float) (buf[index] + buf[index + 2]);
    out[1] = (float) (buf[index + 1] + buf[index + 3]);
    out[2] = (float) (buf[index + 4] + buf[index + 6]);
    out[3] = (float) -(buf[index + 5] + buf[index + 7]);
}

void convertUint8ToFloat(const void *__restrict__ in, const uint32_t index,
                         float *__restrict__ out) {

    const uint8_t *buf = (uint8_t *) in;

    out[0] = (float) (buf[index] + buf[index + 2] - 254);       // ar
    out[1] = (float) (254 - buf[index + 1] - buf[index + 3]);   // aj
    out[2] = (float) (buf[index + 4] + buf[index + 6] - 254);   // br
    out[3] = (float) (buf[index + 5] + buf[index + 7] - 254);   // bj
}

float fastRsqrt(float y) {

    static union fastRsqrtPun pun;

    pun.f = y;
    pun.i = -(pun.i >> 1) + 0x5f3759df;
    pun.f *= 0.5f * (-y * pun.f * pun.f + 3.f);

    return pun.f;
}

float slowRsqrt(float y) {
    return 1.f/sqrtf(y);
}

void fmDemod(const void *__restrict__ buf, const uint32_t len, float *__restrict__ result,
             conversionFunction_t convert, fastRsqrtFun_t rsqrt) {

    static float out[4] = {0.f, 0.f, 0.f, 0.f};
    static float *ar = out;
    static float *aj = out + 1;
    static float *br = out + 2;
    static float *bj = out + 3;

    uint32_t i;
    float zr, zj, y;

    for (i = 0; i < len; i++) {

        convert(buf, i, out);

        zr = fmaf(*ar, *br, -*aj * *bj);
        zj = fmaf(*ar, *bj, *aj * *br);
        y = rsqrt(fmaf(zr, zr, zj * zj));

        zr = 64.f * zj * y * 1.f / fmaf(zr * y, 23.f, 41.f);

        result[i >> 2] = isnan(zr) ? 0.f : zr;
    }
}

int processMatrix(float squelch, FILE *inFile, struct chars *chars, void *outFile) {

    const size_t shiftedSize = DEFAULT_BUF_SIZE - 2;

    void *buf = calloc(DEFAULT_BUF_SIZE, INPUT_ELEMENT_BYTES);
    int exitFlag = 0;
    size_t readBytes;
    float result[QTR_BUF_SIZE];

    while (!exitFlag) {

        readBytes = fread(buf + 2, INPUT_ELEMENT_BYTES, shiftedSize, inFile);

        if ((exitFlag = ferror(inFile))) {
            perror(NULL);
            break;
        } else if (feof(inFile)) {
            exitFlag = EOF;
        }

        fmDemod(buf, readBytes, result, convertUint8ToFloat, fastRsqrt);
//        fmDemod(buf, readBytes, result, convertInt16ToFloat, fastRsqrt);
//        fmDemod(buf, readBytes, result, noconversion, slowRsqrt);

        fwrite(result, OUTPUT_ELEMENT_BYTES, readBytes >> 2, outFile);
    }
    return exitFlag;
}
