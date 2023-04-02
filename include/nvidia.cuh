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
int8_t processMatrix(float squelch, FILE *inFile, struct chars *chars, FILE *outFile);
#endif //DEMODULATOR_NVIDIA_CUH
