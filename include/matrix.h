//
// Created by peads on 4/4/23.
//

#ifndef DEMODULATOR_MATRIX_H
#define DEMODULATOR_MATRIX_H

#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "definitions.h"

struct chars {
    uint8_t isRdc;      // 0
    uint8_t isOt;       // 1
    //uint8_t downsample; // 2
};

int8_t processMatrix(float squelch, FILE *inFile, struct chars *chars, FILE *outFile);
void fmDemod(const uint8_t *buf, uint32_t len, float *result);

#endif //DEMODULATOR_MATRIX_H
