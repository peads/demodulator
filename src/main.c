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
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include "matrix.h"

#ifdef IS_NVIDIA
extern int processMatrix(FILE *__restrict__ inFile, uint8_t mode, float inGain, void *__restrict__ outFile);
#endif

int printIfError(FILE *file) {

    if (!file) {
        perror(NULL);
        return 1;
    }
    return 0;
}

int main(int argc, char **argv) {

    float gain = 1.f;
    uint8_t mode = 0;
    int ret = 0;
    int opt;
    FILE *inFile = NULL;
#if defined(IS_INTEL) || defined(IS_ARM)
    char *outFile = NULL;
#else
    FILE *outFile = NULL;
#endif

    if (argc < 3) {
        return -1;
    } else {
        while ((opt = getopt(argc, argv, "g:i:o:r:")) != -1) {
            switch (opt) {
                case 'g':
                    gain = strtof(optarg, NULL);
                    break;
                case 'r' :
                    mode |= 0b11 & atoi(optarg);
                    break;
                case 'i':
                    if (!strstr(optarg, "-")) {
                        ret += printIfError(inFile = fopen(optarg, "rb"));
                    } else {
                        ret += printIfError(freopen(NULL, "rb", stdin));
                        inFile = stdin;
                    }
                    break;
                case 'o':
#if defined(IS_INTEL) || defined(IS_ARM)
                    outFile = !strstr(optarg, "-") ? optarg : NULL;
#else
                    if (!strstr(optarg, "-")) {
                        ret += printIfError(outFile = fopen(optarg, "wb"));
                    } else {
                        ret += printIfError(freopen(NULL, "wb", stdout));
                        outFile = stdout;
                    }
#endif
                    break;
                default:
                    break;
            }
        }
    }

    if (!ret) {
        ret = processMatrix(inFile, mode, gain, outFile);
#if defined(IS_INTEL) || defined(IS_ARM)
        }
#else
        ret = ret != EOF;
        fclose(outFile);
#endif
        fclose(inFile);
    }
    return ret;
}
