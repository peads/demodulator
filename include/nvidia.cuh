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
#define BLOCKDIM 256
#undef DEFAULT_BUF_SIZE
#define DEFAULT_BUF_SIZE 4194304
struct chars {
    uint8_t isRdc;      // 0
    uint8_t isOt;       // 1
    //uint8_t downsample; // 2
};
#endif //DEMODULATOR_NVIDIA_CUH
