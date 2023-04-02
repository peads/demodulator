//
// Created by peads on 4/2/23.
//

#ifndef DEMODULATOR_NVIDIA_CUH
#define DEMODULATOR_NVIDIA_CUH
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <unistd.h>
#include <cstdlib>
#include "definitions.h"
struct chars {
    uint8_t isRdc;      // 0
    uint8_t isOt;       // 1
    //uint8_t downsample; // 2
};
int8_t processMatrix(float squelch, FILE *inFile, struct chars *chars, FILE *outFile);
#endif //DEMODULATOR_NVIDIA_CUH
