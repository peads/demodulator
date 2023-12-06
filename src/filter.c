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

#include "filter.h"

inline LREAL warpButter(const LREAL alpha,
                 const LREAL beta,
                 const size_t k,
                 const size_t n,
                 LREAL *__restrict__ z) {

    size_t j = (k - 1) << 2;
    const LREAL w = M_PI_2 * (1. / (LREAL) n * (-1. + (LREAL) (k << 1)) + 1.);
    const LREAL a = COS(w);
    const LREAL d = 1. / (a - alpha);// 1. / SIN(2. * theta));
    const LREAL zr = (a - beta/*tan(theta)*/) * d;
    const LREAL zj = SIN(w) * d;

    z[j + 2] = z[j] = 1. - zr;
    z[j + 1] = zj;
    z[j + 3] = -zj;

    return zr;
}

inline LREAL warpCheby1(const LREAL tng,
                 const LREAL ep,
                 const size_t k,
                 const size_t n,
                 LREAL *__restrict__ z) {

    size_t j = (k - 1) << 2;
    const LREAL oneOverN = 1. / (LREAL) n;
    const LREAL v = LOG((1. + POW(10., 0.5 * ep)) / SQRT(POW(10., ep) - 1.)) * oneOverN;
    const LREAL t = M_PI_2 * (oneOverN * (-1. + (LREAL) (k << 1)));

    const LREAL a = COS(t) * COSH(v) * tng; // tan(w * wh)
    const LREAL b = SIN(t) * SINH(v) * tng;
    const LREAL c = a * a + b * b;
    const LREAL d = 1. / (1. + c + 2. * b);
    LREAL zj = 2. * a * d;
    LREAL zr = 2. * (b + c) * d;

    z[j + 2] = z[j] = 1. - zr;
    z[j + 1] = zj;
    z[j + 3] = -zj;

    return zr;
}

static inline void generateCoeffs(const size_t k,
                                  const size_t n,
                                  const LREAL alpha,
                                  const LREAL beta,
                                  const warpGenerator_t warp,
                                  LREAL *__restrict__ acc,
                                  LREAL *__restrict__ z) {

    LREAL a, zj;
    LREAL zr = warp(alpha, beta, k, n, z);
    zj = z[((k - 1) << 2) + 1]; // 2k - 1

    if (k <= n >> 1) {
        a = zr * zr + zj * zj;
        acc[0] *= a;
        acc[1] *= a;
    } else {
        a = zr * acc[0] - zj * acc[1];
        acc[1] = zr * acc[1] + zj * acc[0];
        acc[0] = a;
    }
#ifdef VERBOSE
    fprintf(stderr, PRINT_POLES, 1. - zr, zj);
#endif
}

/// Note this simplification will not work for non-bilinear transform transfer functions
static inline void zp2Sos(const size_t n, const LREAL *z, const LREAL *p, const LREAL k, LREAL sos[][6]) {

    size_t i, j;
    size_t npc = n >> 1;
    size_t npr = 0;

    if (n & 1) {
        npr = 1;
    }

    for (j = 0, i = 0; j < npc; i += 4, ++j) {
        sos[j][3] = sos[j][0] = 1.;
        sos[j][1] = -2. * z[i];
        sos[j][2] = z[i] * z[i] + z[i + 1] * z[i + 1];
        sos[j][4] = -2. * p[i];
        sos[j][5] = p[i] * p[i] + p[i + 1] * p[i + 1];
    }

//    for (j = npc, i = (n << 1) - npc + 1; j < npc + npr; i += 4, ++j) {
    if (npc < npc + npr) {
        sos[npc][3] = 1.;
        sos[npc][2] = sos[npc][5] = 0.;
        sos[npc][0] = sos[npc][1] = k;
        sos[npc][4] = -p[(n << 1) - 2];
    }
}

inline LREAL transformBilinear(const size_t n,
                         const LREAL a,
                         const LREAL b,
                         LREAL sos[][6],
                         const warpGenerator_t warp) {

    size_t i, k;
    LREAL acc[2] = {1., 0};
    LREAL *p = calloc(((n + 1) << 1), sizeof(LREAL));
    LREAL *z = calloc(((n + 1) << 1), sizeof(LREAL));
    LREAL *t = calloc((n << 1), sizeof(LREAL));
    size_t N = n >> 1;
    N = (n & 1) ? N + 1 : N;
#ifdef VERBOSE
    fprintf(stderr, "\nz: There are n = %zu zeros at z = -1 for (z+1)^n\np: ", n);
#endif
    // Generate roots of bilinear transform
    for (k = 1; k <= N; ++k) {
        generateCoeffs(k, n, a, b, warp, acc, p);
    }

    // TODO correct K for even degrees
    // Store the gain
    acc[0] /= POW(2., (LREAL) n);

    for (i = 0; i < n << 1; i += 2) {
        z[i] = -1.;
        z[i + 1] = 0;
    }

    zp2Sos(n, z, p, acc[0], sos);

#ifdef VERBOSE
    size_t j;
    k = n >> 1;
    k = (n & 1) ? k + 1 : k;
    fprintf(stderr, PRINT_K, acc[0]);
    for (i = 0; i < k; ++i) {
        for (j = 0; j < 6; ++j) {
            fprintf(stderr, PRINT_SOS, sos[i][j]);
        }
        fprintf(stderr, "\n");
    }
#endif

    free(p);
    free(t);
    free(z);
    return acc[0];
}

inline void applyFilter(REAL *__restrict__ x,
                 REAL *__restrict__ y,
                 const size_t len,
                 const size_t sosLen,
                 const REAL sos[][6],
                 const windowGenerator_t wind) {

    REAL *xp, *yp, c[2];
    size_t i, j, m;

    for (i = 0; i < len; ++i) {
        j = i + (sosLen >> 1);//sosLen;
        xp = &x[j];
        yp = &y[j];
        for (m = 0; m < sosLen; ++m) {

            c[0] = wind(m, sosLen + 1);
            c[1] = wind(m + 1, sosLen + 1);

            yp[m] = sos[m][0] * yp[m] + sos[m][1] * yp[m + 1] + 1;
            yp[m] -= sos[m][3] + sos[m][4] * c[0] * xp[m] + sos[m][5] * c[1] * xp[m + 1];
        }
    }
}

inline void applyComplexFilter(REAL *__restrict__ x,
                        REAL *__restrict__ y,
                        const size_t len,
                        const size_t sosLen,
                        const REAL sos[][6],
                        const windowGenerator_t wind) {

    REAL *xp, *yp, c[2] = {};
    size_t i, l, j, m;

    for (i = 0; i < len; i += 2) {

        j = i + sosLen;//(sosLen << 1);
        yp = &y[j];
        xp = &x[j];

        for (m = 0; m < sosLen; ++m) {

            l = m << 1;

            c[0] = wind(m, sosLen + 1);
            c[1] = wind(m + 1, sosLen + 1);

            yp[l] = sos[m][0] * yp[l] + sos[m][1] * yp[l + 2] + 1;
            yp[l] -= sos[m][3] + sos[m][4] * c[0] * xp[l] + sos[m][5] * c[1] * xp[l + 2];

            yp[l + 1] = sos[m][0] * yp[l + 1] + sos[m][1] * yp[l + 3];
            yp[l + 1] -= sos[m][3] + sos[m][4] * c[0] * xp[l + 1] + sos[m][5] * c[1] * xp[l + 3];
        }
    }
}


