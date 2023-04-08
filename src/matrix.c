//
// Created by peads on 4/3/23.
//
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
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

//        result[i >> 2] = atan2f(zj, zr);
        lenR = 1.f/sqrtf(fmaf(zr, zr, zj*zj));
        zr = 64.f*zj*lenR * 1.f/fmaf(zr*lenR, 23.f, 41.f);

        result[i >> 2] = isnan(zr) ? 0.f : zr;
    }
}


int8_t processMatrix(float squelch, FILE *inFile, struct chars *chars, FILE *outFile) {

    uint8_t *buf = calloc(DEFAULT_BUF_SIZE, INPUT_ELEMENT_BYTES);
    int8_t exitFlag = 0;
    size_t readBytes;
    float result[QTR_BUF_SIZE];

    readBytes = fread(buf + 2, INPUT_ELEMENT_BYTES, DEFAULT_BUF_SIZE - 2, inFile);

    while (!exitFlag) {

        if (exitFlag = ferror(inFile)) {
            perror(NULL);
            break;
        } else if (feof(inFile)) {
            exitFlag = EOF;
        }

        fmDemod(buf, readBytes, result);

        fwrite(result, OUTPUT_ELEMENT_BYTES, QTR_BUF_SIZE, outFile);

        readBytes = fread(buf, INPUT_ELEMENT_BYTES, DEFAULT_BUF_SIZE, inFile);
    }
    return exitFlag;
}
