/*
 * This file is part of the demodulator distribution
 * (https://github.com/peads/demodulator).
 * with code originally part of the misc_snippets distribution
 * (https://github.com/peads/misc_snippets).
 * Copyright (c) 2023 Patrick Eads.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef DEMODULATOR_MATRIX_H
#define DEMODULATOR_MATRIX_H
#include <stdint.h>
#include <math.h>
#include <pthread.h>
#include <semaphore.h>
#include <string.h>
#include "prototypes.h"
#include "definitions.h"
#ifdef HAS_AVX
#include <immintrin.h>
#endif
#if __GNUC__ < 10
#include <math.h>
#include <stdint.h>
#endif
#ifdef __INTEL_COMPILER
#include <stdlib.h>
#endif

typedef struct {
    sem_t full, empty;
    const uint8_t mode;
    void *buf;
    int exitFlag;
    FILE *outFile;
    pthread_mutex_t mutex;
    float gain;
} consumerArgs;
#ifdef HAS_AVX
typedef union {
    __m256i v;
    int8_t buf[32];
} m256i_pun_t;
#endif
#ifdef HAS_AVX512
typedef union {
    __m512i v;
    int8_t buf[64];
} m512i_pun_t;
#endif
typedef void (*conversionFunction_t)(const void *__restrict__, const uint32_t, float *__restrict__);
#endif //DEMODULATOR_MATRIX_H
