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

extern uint64_t filter(__m128 *buf, uint64_t len, uint8_t downsample);
extern void removeDCSpike(__m128 *buf, uint64_t len);
extern void applyComplexConjugate(__m128 *buf, uint64_t len);
extern uint64_t demodulateFmData(__m128 *buf, uint64_t len, uint64_t *result);

__attribute((used)) static void checkFileStatus(FILE *file) {

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

void processMatrix(FILE *inFile, FILE *outFile, uint8_t downsample, uint8_t isRdc, uint8_t isOt, __m128 *squelch) {

    union {
        uint8_t buf[MATRIX_WIDTH];
        __m128 u;
    } z;

    uint64_t j;
    uint64_t depth;
    uint64_t len;
    __m128 buf[DEFAULT_BUF_SIZE];
    uint64_t result[DEFAULT_BUF_SIZE];

    while (!exitFlag) {
        for(j = 0, len = 0; j < DEFAULT_BUF_SIZE; ++j) {
//            len += fread(z.buf, INPUT_ELEMENT_BYTES, MATRIX_WIDTH, inFile);
//            checkFileStatus(inFile);
            __asm__ (
                    "leaq (%5), %%rdi\n\t"
                    "movq $1, %%rsi\n\t"
                    "movq $4, %%rdx\n\t"
                    "leaq (%4), %%rcx\n\t"
                    "call _fread\n\t"
                    "addq %%rax, %0\n\t"
//                    "xorq %1, %1\n\t"
//                    "xorq %%rax, %%rax\n\t"
//                    "movq $1024, %1\n\t"
//                    "negq %1\n\t"
//                "L6: "
//                    "addq %%rax, %2\n\t"

                    "vpaddb all_nonetwentysevens(%%rip), %2, %1\n\t"
                    "vpmovsxbw %1, %1\n\t"
                    "vpmovsxwd %1, %1\n\t"
                    "vcvtdq2ps %1, %1\n\t"
                    "orq %3, %3\n\t"                    // if squelch != NULL
                    "jz nosquelch\n\t"                  // apply squelch
                    "vmulps %1, %1, %%xmm2\n\t"
                    "vpermilps $0xB1, %%xmm2, %%xmm3\n\t"
                    "vaddps %%xmm2, %%xmm3, %%xmm2\n\t"
                    "vmulps all_hundredths(%%rip), %%xmm2, %%xmm2\n\t"
                    "vcmpps $0x1D, (%3), %%xmm2, %%xmm2\n\t"
                    "vandps %%xmm2, %1, %1\n\t"
                "nosquelch:\n\t"
//                    "leaq (%%rcx), %%rdi\n\t"
//                    "callq _checkFileStatus\n\t"
//                    "add $1, %1\n\t"
//                    "jl L6\n\t"
                    :"+r"(len), "=x"(buf[j]) : "x"(z.u), "r"(squelch), "r"(inFile), "r"(z.buf) : "rdi", "rsi", "rdx", "rcx", "xmm2", "xmm3");
        }

        if (!len) break;
        if (len) {
            if (isRdc) {
                removeDCSpike(buf, DEFAULT_BUF_SIZE);
            }

            if (!isOt) {
                applyComplexConjugate(buf, DEFAULT_BUF_SIZE);
            }

            depth = filter(buf, DEFAULT_BUF_SIZE, downsample);
            depth = demodulateFmData(buf, depth, result);

            fwrite(result, OUTPUT_ELEMENT_BYTES, depth, outFile);
        }
    }
}

int main(int argc, char **argv) {

    int opt;
    uint8_t downsample;
    uint8_t isRdc = 0;
    uint8_t isOt = 0;
    __m128 *squelch = NULL;
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
                    squelch = malloc(sizeof(__m128));
                    *squelch = _mm_set1_ps(powf(10.f, (float) atof(optarg) / 10.f));
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
    if (squelch) free(squelch);

    return exitFlag != EOF;
}
