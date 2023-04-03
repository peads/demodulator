//
// Created by peads on 4/3/23.
//
#include <stdio.h>
#include "definitions.h"
#include "matrix.h"
#include "math.h"

void fmDemod(const uint8_t *buf, const uint32_t len, float *result) {

    uint32_t i;
    float ar, aj, br, bj, zr, zj;

    for (i = 0; i < len; i++) {

        ar = (float)(buf[i  ] + buf[i+2] - 254);
        aj = (float)(254 - buf[i+1] - buf[i+3]);

        br = (float)(buf[i+4] + buf[i+6] - 254);
        bj = (float)(buf[i+5] + buf[i+7] - 254);

        zr = ar*br - aj*bj;//__fmaf_rz(ar, br, -__fmul_rz(aj, bj));
        zj = ar*bj + aj*br;//__fmaf_rz(ar, bj, __fmul_rz(aj, br));

        result[i >> 2] = atan2f(zj, zr);
    }
}


int8_t processMatrix(float squelch, FILE *inFile, struct chars *chars, FILE *outFile) {

    uint8_t buf[DEFAULT_BUF_SIZE];
    float result[QTR_BUF_SIZE];

    int8_t exitFlag = 0;
    size_t readBytes;

    while (!exitFlag) {

        readBytes = fread(buf, INPUT_ELEMENT_BYTES, DEFAULT_BUF_SIZE, inFile);

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
