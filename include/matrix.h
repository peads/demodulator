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
#include <math.h>
#include <pthread.h>
#include <semaphore.h>
#include <string.h>
#include <stdint.h>
#include "filter.h"
#if __GNUC__ < 10
#include <math.h>
#include <stdint.h>
#endif
#ifdef __INTEL_COMPILER
#include <stdlib.h>
#endif

#ifndef DEFAULT_BUF_SIZE
#define DEFAULT_BUF_SIZE 262144L
#endif

typedef void (*iqCorrection_t)(void *__restrict__, size_t, REAL *__restrict__);

typedef struct {
    sem_t *full, *empty;
    pthread_mutex_t mutex;
    FILE *outFile;
    void *buf;
    int exitFlag;
    uint8_t mode;
    LREAL sampleRate;
    LREAL lowpassIn;
    LREAL lowpassOut;
    LREAL epsilon;
    size_t inFilterDegree;
    size_t outFilterDegree;
    size_t bufSize;
} consumerArgs;

#ifndef IS_NVIDIA
void *processMatrix(void *ctx);
void allocateBuffer(void **buf, size_t len);
#endif

#endif //DEMODULATOR_MATRIX_H
