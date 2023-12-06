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
#ifndef DEMODULATOR_FILTER_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#define REAL float

#ifdef PRECISION
    #undef REAL
    #define REAL double
#endif

#if PRECISION && DBL_MANT_DIG < LDBL_MANT_DIG
    #define LREAL long double
    #define LOG logl
    #define POW powl
    #define SQRT sqrtl
    #define COS cosl
    #define SIN sinl
    #define TAN tanl
    #define COSH coshl
    #define SINH sinhl
    #define ACOSH acoshl
    #define PRINT_POLES "(%Lf +/- %Lf I) "
    #define PRINT_K "\nk: %Le\n"
    #define PRINT_SOS "%Lf "
    #define PRINT_EP_WC "\nepsilon: %Lf\nwc: %Lf"
    #define TO_REAL strtold
#else
    #define LREAL double
    #define LOG log
    #define POW pow
    #define SQRT sqrt
    #define COS cos
    #define SIN sin
    #define TAN tan
    #define COSH cosh
    #define SINH sinh
    #define ACOSH acosh
    #define PRINT_POLES "(%f +/- %f I) "
    #define PRINT_K "\nk: %e\n"
    #define PRINT_SOS "%f "
    #define PRINT_EP_WC "\nepsilon: %f\nwc: %f"
    #define TO_REAL strtod
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
