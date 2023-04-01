//
// Created by peads on 4/1/23.
//

#ifndef DEMODULATOR_MATRIX_CUH
#define DEMODULATOR_MATRIX_CUH
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include "definitions.h"

struct chars {
    uint8_t isRdc;      // 0
    uint8_t isOt;       // 1
    //uint8_t downsample; // 2
};

//__global__ void readFile(float squelch, FILE *inFile, struct chars *chars, FILE *outFile);
#endif //DEMODULATOR_MATRIX_CUH
