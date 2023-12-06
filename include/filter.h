//
// Created by peads on 12/5/23.
//
#ifndef DEMODULATOR_FILTER_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#define REAL float
#define LREAL double

#ifdef PRECISION
    #undef REAL
    #define REAL double
#endif

#if PRECISION && DBL_MANT_DIG < LDBL_MANT_DIG
    #undef LREAL
    #define LREAL long double
#endif

typedef REAL  (*windowGenerator_t)(size_t, size_t);
typedef LREAL (*warpGenerator_t)(LREAL, LREAL, size_t, size_t, LREAL *__restrict__);

/* Filter digitization interfaces */
LREAL warpButter(LREAL, LREAL, size_t, size_t, LREAL *__restrict__);
LREAL warpCheby1(LREAL, LREAL, size_t, size_t, LREAL *__restrict__);
LREAL transformBilinear(size_t, LREAL, LREAL, LREAL[][6], warpGenerator_t);

/* Filter application interfaces */
void applyFilter(
        REAL *__restrict__, REAL *__restrict__, size_t, size_t, const REAL[][6], windowGenerator_t);

void applyComplexFilter(
        REAL *__restrict__, REAL *__restrict__, size_t, size_t, const REAL[][6], windowGenerator_t);

#define DEMODULATOR_FILTER_H

#endif //DEMODULATOR_FILTER_H
