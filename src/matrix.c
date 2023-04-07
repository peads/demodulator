//
// Created by peads on 4/3/23.
//
#include <stdio.h>
#include <math.h>
#include "definitions.h"
#include "matrix.h"


void fmDemod(const uint8_t *__restrict__ buf, const uint32_t len, float *__restrict__ result) {

    uint32_t i;
    float ar, aj, br, bj, zr, zj, lenR;

    for (i = 0; i < len; i++) {

        ar = (float)(buf[i  ] + buf[i+2] - 254);
        aj = (float)(254 - buf[i+1] - buf[i+3]);

        br = (float)(buf[i+4] + buf[i+6] - 254);
        bj = (float)(buf[i+5] + buf[i+7] - 254);

        zr = fmaf(ar, br, -aj*bj);
        zj = fmaf(ar, bj, aj*br);

        lenR = 1.f/sqrtf(fmaf(zr, zr, zj*zj));
        zr = 64.f*zj*lenR * 1.f/fmaf(zr*lenR, 23.f, 41.f);

        result[i >> 2] = isnan(zr) ? 0.f : zr;
    }
}


int8_t processMatrix(float squelch, FILE *inFile, struct chars *chars, FILE *outFile) {

    static const uint32_t len = (DEFAULT_BUF_SIZE + 2);

    uint8_t buf[len];
    uint8_t prevR = 0;
    uint8_t prevJ = 0;
    int8_t exitFlag = 0;
    size_t readBytes;
    float result[QTR_BUF_SIZE];

    while (!exitFlag) {

        buf[0] = prevR;
        buf[1] = prevJ;
        readBytes = fread(buf + 2, INPUT_ELEMENT_BYTES, DEFAULT_BUF_SIZE, inFile);
        prevR = buf[len - 2];
        prevJ = buf[len - 1];

        if (exitFlag = ferror(inFile)) {
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
