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

#include "matrix.h"

void fmDemod1(const uint8_t *__restrict__ buf, const uint32_t len, float *__restrict__ result) {

    uint32_t i;
    int32_t ar, aj, br, bj, zr, zj;

    for (i = 0; i < len; i++) {

        ar = buf[i] + buf[i+2] - 254;
        aj = 254 - buf[i+1] - buf[i+3];

        br = buf[i+4] + buf[i+6] - 254;
        bj = buf[i+5] + buf[i+7] - 254;

        zr = ar*br - aj*bj;
        zj = ar*bj + aj*br;

        result[i >> 2] = atan2f(zj, zr);
    }
}