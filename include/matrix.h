//
// Created by peads on 4/3/23.
//

#ifndef DEMODULATOR_MATRIX_H
#define DEMODULATOR_MATRIX_H
#include <stdint.h>

struct chars {
    uint8_t isRdc;      // 0
    uint8_t isOt;       // 1
    //uint8_t downsample; // 2
};

union fastSqrtPun {
    uint32_t i;
    float f;
};


#endif //DEMODULATOR_MATRIX_H
