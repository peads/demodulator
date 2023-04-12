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
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <stdlib.h>
#include <immintrin.h>
#include "matrix.h"

__attribute__((used)) uint64_t result[DEFAULT_BUF_SIZE] __attribute__((aligned(16)));
__attribute__((used)) __m128 temp;
__attribute__((used)) __m128 temp1;

extern int8_t processMatrix(float squelch, FILE *inFile, struct chars *chars, FILE *outFile);

int main(int argc, char **argv) {

    int opt;
    float temp = 0.f;
    FILE *inFile = NULL;
    FILE *outFile = NULL;
    struct chars chars;
    chars.isOt = 0;
    chars.isRdc = 0;

    if (argc < 3) {
        return -1;
    } else {
        while ((opt = getopt(argc, argv, "i:o:s:rf")) != -1) {
            switch (opt) {
                case 'r':
                    chars.isRdc = 1;
                    break;
                case 'f':
                    chars.isOt = 1;
                    break;
//                case 'd': // TODO reimplement downsmapling?
//                    chars.downsample = atoi(optarg);
//                    break;
                case 's':   // TODO add parameter to take into account the impedance of the system
                            // currently calculated for 50 Ohms (i.e. Prms = ((I^2 + Q^2)/2)/50 = (I^2 + Q^2)/100)
                    temp = powf(10.f, (float) atof(optarg) / 10.f);
                    break;
                case 'i':
                    if (!strstr(optarg, "-")) {
                        inFile = fopen(optarg, "rb");
                    } else {
                        freopen(NULL, "rb", stdin);
                        inFile = stdin;
                    }
                    break;
                case 'o':
                    if (!strstr(optarg, "-")) {
                        outFile = fopen(optarg, "wb");
                    } else {
                        freopen(NULL, "wb", stdout);
                        outFile = stdout;
                    }
                    break;
                default:
                    break;
            }
        }
    }
    return processMatrix(temp, inFile, &chars, outFile) != EOF;
}
