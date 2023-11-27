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
    sem_t *full, *empty;
    void *buf;
    int exitFlag;
    FILE *outFile;
    pthread_mutex_t mutex;
    float sampleRate;
    float lowpassIn;
    float lowpassOut;
    size_t inFilterDegree;
    size_t outFilterDegree;
    float epsilon;
    uint8_t mode;
    size_t bufSize;
} consumerArgs;

#ifdef HAS_AVX512
static const __m512 ALL_64S = {
        64.f, 64.f, 64.f, 64.f, 64.f, 64.f, 64.f, 64.f,
        64.f, 64.f, 64.f, 64.f, 64.f, 64.f, 64.f, 64.f};
static const __m512 ALL_23S = {
        23.f, 23.f, 23.f, 23.f, 23.f, 23.f, 23.f, 23.f,
        23.f, 23.f, 23.f, 23.f, 23.f, 23.f, 23.f, 23.f};
static const __m512 ALL_41S = {
        41.f, 41.f, 41.f, 41.f, 41.f, 41.f, 41.f, 41.f,
        41.f, 41.f, 41.f, 41.f, 41.f, 41.f, 41.f, 41.f};
static const __m512 MUL_CONJ = {
        1, 1, -1.f, 1, 1, 1, -1.f, 1,
        1, 1, -1.f, 1, 1, 1, -1.f, 1};
static const __m512i ORIGIN_SHIFT_UINT8 = {
        //_mm512_set1_epi8(-127);
        -0x7e7e7e7e7e7e7e7f,
        -0x7e7e7e7e7e7e7e7f,
        -0x7e7e7e7e7e7e7e7f,
        -0x7e7e7e7e7e7e7e7f,
        -0x7e7e7e7e7e7e7e7f,
        -0x7e7e7e7e7e7e7e7f,
        -0x7e7e7e7e7e7e7e7f,
        -0x7e7e7e7e7e7e7e7f};
static const __m512 ONES = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
static const __m512 alpha = {
        0.99212598425f,1,0.99212598425f,1,0.99212598425f,1,0.99212598425f,1,
        0.99212598425f,1,0.99212598425f,1,0.99212598425f,1,0.99212598425f,1};
static const __m512 beta = {
        0,0.00787401574f,0,0.00787401574f,0,0.00787401574f,0,0.00787401574f,
        0,0.00787401574f,0,0.00787401574f,0,0.00787401574f,0,0.00787401574f};
#elif defined(HAS_AVX)
static const __m256 ALL_64S = {
        64.f, 64.f, 64.f, 64.f,
        64.f, 64.f, 64.f, 64.f};
static const __m256 ALL_23S = {
        23.f, 23.f, 23.f, 23.f,
        23.f, 23.f, 23.f, 23.f};
static const __m256 ALL_41S = {
        41.f, 41.f, 41.f, 41.f,
        41.f, 41.f, 41.f, 41.f};
static const __m256i INDEX_FM_DEMOD_ORDERING = {
        // _mm256_setr_epi32(1, 3, 5, 7, 0, 2, 3, 6);
        0x300000001,
        0x700000005,
        0x200000000,
        0x600000003};
static const __m256i INDEX_CONJ_ORDERING = {
        // _mm256_setr_epi32(0,2,4,6,1,3,5,7);
        0x200000000,
        0x600000004,
        0x300000001,
        0x700000005};
static const __m256 MUL_CONJ = {1, 1, -1.f, 1, 1, 1, -1.f, 1};
// Degree 8 coefficients, negated to efficiently subtract
static const __m256 ONES = {1,1,1,1,1,1,1,1};
static const __m256i ORIGIN_SHIFT_UINT8 = {
        //_mm256_set1_epi8(-127);
        -0x7e7e7e7e7e7e7e7f,
        -0x7e7e7e7e7e7e7e7f,
        -0x7e7e7e7e7e7e7e7f,
        -0x7e7e7e7e7e7e7e7f};
static const __m256 alpha = {0.99212598425f,1,0.99212598425f,1,0.99212598425f,1,0.99212598425f,1};
static const __m256 beta = {0,0.00787401574f,0,0.00787401574f,0,0.00787401574f,0,0.00787401574f};
#endif
#endif //DEMODULATOR_MATRIX_H
