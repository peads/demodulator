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

#include "demodulator.h"

//extern uint64_t filter(__m128 *buf, uint64_t len, uint8_t downsample);
//extern void removeDCSpike(__m128 *buf, uint64_t len);
//extern void applyComplexConjugate(__m128 *buf, uint64_t len);
//extern uint64_t demodulateFmData(__m128 *buf, uint64_t len, uint64_t *result);

__attribute__((used)) void checkFileStatus(FILE *file) {

    if (ferror(file)) {
        char errorMsg[256];
        sprintf(errorMsg, "\nI/O error when reading file");
        perror(errorMsg);
        exitFlag = 1;
    } else if (feof(file)) {
#ifdef DEBUG
        fprintf(stderr, "\nExiting\n");
#endif
        exitFlag = EOF;
    }
}
                        //rdi           rsi         xmm0            rdx             rcx            r8                  r9
extern void readFile(uint8_t *buf, uint64_t len, __m128 squelch, __m128 *buf128, FILE *inFile, struct chars *chars, FILE *outFile);

void processMatrix(FILE *inFile, FILE *outFile, uint8_t downsample, uint8_t isRdc, uint8_t isOt, __m128 squelch) {

//    uint64_t depth = 0;
//    uint64_t ret = 0;
    uint8_t buf[MATRIX_WIDTH] __attribute__((aligned (16)));
    struct chars chars;

    chars.isRdc = isRdc;
    chars.isOt = isOt;
    chars.downsample = downsample;
    readFile(buf, DEFAULT_BUF_SIZE, squelch, buf128, inFile, &chars, outFile);
    printf("lol\n");
//    while (1) {
//
//        ret = readFile(buf, DexitFlagAULT_BUF_SIZE, squelch, buf128, inFile, &chars, outFile);
//
//        if (exitFlag) {
//            break;
//        }
//
//        if (ret) {
//            if (isRdc) {
//                removeDCSpike(buf128, DexitFlagAULT_BUF_SIZE);
//            }
//
//            if (!isOt) {
//                applyComplexConjugate(buf128, DexitFlagAULT_BUF_SIZE);
//            }
//
//            depth = filter(buf128, DexitFlagAULT_BUF_SIZE, downsample);
//            depth = demodulateFmData(buf128, depth, result);
//
//            fwrite(result, OUTPUT_ELEMENT_BYTES, depth, outFile);
//        }
//    }
}

int main(int argc, char **argv) {

    int opt;
    uint8_t downsample;
    uint8_t isRdc = 0;
    uint8_t isOt = 0;
    __m128 squelch = {0,0,0,0};
    FILE *inFile = NULL;
    FILE *outFile = NULL;

    if (argc < 3) {
        return -1;
    } else {
        while ((opt = getopt(argc, argv, "i:o:d:s:rf")) != -1) {
            switch (opt) {
                case 'r':
                    isRdc = 1;
                    break;
                case 'f':
                    isOt = 1;
                    break;
                case 'd':
                    downsample = atoi(optarg);
                    break;
                case 's':   // TODO add parameter to take into account the impedance of the system
                            // currently calculated for 50 Ohms (i.e. Prms = ((I^2 + Q^2)/2)/50 = (I^2 + Q^2)/100)
                    squelch = _mm_set1_ps(powf(10.f, (float) atof(optarg) / 10.f));
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

    processMatrix(inFile, outFile, downsample, isRdc, isOt, squelch);

    fclose(outFile);
    fclose(inFile);

    return exitFlag != EOF;
}
