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
#include "definitions.h"
#include "matrix.h"
#include "fmath.h"

conversionFunction_t convert;

static void convertInt16ToFloat(const void *__restrict__ in, const uint32_t index,
                                float *__restrict__ out) {

    const int16_t *buf = (int16_t *) in;

    out[0] = (float) (buf[index] + buf[index + 2]);
    out[1] = (float) -(buf[index + 1] + buf[index + 3]);
    out[2] = (float) (buf[index + 4] + buf[index + 6]);
    out[3] = (float) (buf[index + 5] + buf[index + 7]);
}

static void convertUint8ToFloat(const void *__restrict__ in, const uint32_t index,
                                float *__restrict__ out) {

    const uint8_t *buf = (uint8_t *) in;
    float magA, magB;
    out[0] = (float) (buf[index] + buf[index + 2] - 254);       // ar
    out[1] = (float) (buf[index + 1] + buf[index + 3] - 254);   // aj
    magA = frsqrtf(fmaf(out[0], out[0],out[1]*out[1]));
    out[0] *= magA;
    out[1] *= magA;
    out[2] = (float) (buf[index + 4] + buf[index + 6] - 254);   // br
    out[3] = (float) -(buf[index + 5] + buf[index + 7] - 254);   // bj
    magB = frsqrtf(fmaf(out[2], out[2],out[3]*out[3]));
    out[2] *= magB;
    out[3] *= magB;
}

static inline void fmDemod(const void *__restrict__ buf,
                           const uint32_t len,
                           const float gain,
                           float *__restrict__ result) {

    static float out[4] = {0.f, 0.f, 0.f, 0.f};

    uint32_t i;
    float zr, zj, ac, bd;

    for (i = 0; i < len; i += 2) {

        convert(buf, i, out);


        ac = out[0] * out[2];
        bd = out[1] * out[3];
        zr = ac - bd;
        zj = fmaf((out[0] + out[1]), (out[2] + out[3]), -(ac + bd));
        zr = 64.f * zj * frcpf(fmaf(23.f, zr, 41.f));

        result[i >> 2] = isnan(zr) ? 0.f : gain ? zr * gain : zr;
    }
}

int processMatrix(FILE *__restrict__ inFile,
                  const uint8_t mode,
                  float gain,
                  void *__restrict__ outFile) {

    convert = !mode ? convertInt16ToFloat : convertUint8ToFloat;
    gain = gain != 1.f ? gain : 0.f;
    int exitFlag = mode < 0 || mode > 1;
    void *buf __attribute__((aligned(32))) = calloc(DEFAULT_BUF_SIZE, 2 - mode);

    size_t readBytes;
    float *result __attribute__((aligned(32))) = calloc(DEFAULT_BUF_SIZE >> 2, sizeof(float));

    while (!exitFlag) {

        readBytes = fread(buf + 2, 2 - mode, DEFAULT_BUF_SIZE - 2, inFile);

        if ((exitFlag = ferror(inFile))) {
            perror(NULL);
            break;
        } else if (feof(inFile)) {
            exitFlag = EOF;
        }

        fmDemod(buf, readBytes, gain, result);

        fwrite(result, OUTPUT_ELEMENT_BYTES, readBytes >> 2, outFile);
    }
    free(buf);
    free(result);
    return exitFlag;
}
