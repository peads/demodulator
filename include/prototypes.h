//
// Created by Patrick Eads on 8/29/23.
//

#ifndef DEMODULATOR_PROTOTYPES_H
#define DEMODULATOR_PROTOTYPES_H
#include <stdint.h>
#include <stdio.h>
#ifndef IS_NVIDIA
int processMatrix(FILE *__restrict__ inFile, uint8_t mode, float gain, void *__restrict__ outFile);
#endif
#endif //DEMODULATOR_PROTOTYPES_H
