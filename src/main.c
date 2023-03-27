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

__attribute__((used)) static void checkFileStatus(FILE *file) {

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
                        //rdi           rsi         rdx         rcx             r8              r9
extern uint64_t readFile(uint8_t *buf, uint64_t len, __m128 *u, __m128 *squelch, __m128 *result, FILE *file);
// TODO reimplement the struct and see if we can obviate all the stack ops with one push and pop of rbx
__asm__(
#ifdef __clang__
"_readFile: "
#else
"readFile: "
#endif
    "pushq %rbp\n\t"
    "movq %rsp, %rbp\n\t"

    "xorq %r11, %r11\n\t"
    "shlq $4, %rsi\n\t"
    "addq %rsi, %r8\n\t"
    "negq %rsi\n\t"
"L6: "
    "pushq %rsi\n\t"    // preserve our registers
    "pushq %rdx\n\t"
    "pushq %rcx\n\t"
    "pushq %rdi\n\t"
    "pushq %r8\n\t"
    "pushq %r9\n\t"
    "movq $1, %rsi\n\t" // set the fread arguments (rdi is implicit)
    "movq $4, %rdx\n\t"
    "leaq (%r9), %rcx\n\t"
    "call _fread\n\t"
    "popq %r9\n\t"
    "popq %r8\n\t"
    "popq %rdi\n\t"
    "popq %rcx\n\t"
    "popq %rdx\n\t"
    "popq %rsi\n\t"
    "addq %rax, %r11\n\t"

    "pushq %rsi\n\t"    // preserve our registers
    "pushq %rdx\n\t"
    "pushq %rcx\n\t"
    "pushq %rdi\n\t"
    "pushq %r8\n\t"
    "pushq %r9\n\t"
    "leaq (%r9), %rdi\n\t"
    "callq _checkFileStatus\n\t"    // TODO consider inlining file err/eof check
    "popq %r9\n\t"
    "popq %r8\n\t"
    "popq %rdi\n\t"
    "popq %rcx\n\t"
    "popq %rdx\n\t"
    "popq %rsi\n\t"

    "vmovaps (%rdi), %xmm1\n\t"
    "vpaddb all_nonetwentysevens(%rip), %xmm1, %xmm1\n\t"
    "vpmovsxbw %xmm1, %xmm1\n\t"
    "vpmovsxwd %xmm1, %xmm1\n\t"
    "vcvtdq2ps %xmm1, %xmm1\n\t"
    "orq %rcx, %rcx\n\t"                // if squelch != NULL
    "jz nosquelch\n\t"                  // apply squelch
    "vmulps %xmm1, %xmm1, %xmm2\n\t"
    "vpermilps $0xB1, %xmm2, %xmm3\n\t"
    "vaddps %xmm2, %xmm3, %xmm2\n\t"
    "vmulps all_hundredths(%rip), %xmm2, %xmm2\n\t"
    "vcmpps $0x1D, (%rcx), %xmm2, %xmm2\n\t"
    "vandps %xmm2, %xmm1, %xmm1\n\t"
"nosquelch:\n\t"
    "vmovaps %xmm1, (%rsi, %r8)\n\t"
    "add $16, %rsi\n\t"
    "jl L6\n\t"

    "popq %rbp\n\t"
    "movq %r11, %rax\n\t"
    "ret"
);

void processMatrix(FILE *inFile, FILE *outFile, uint8_t downsample, uint8_t isRdc, uint8_t isOt, __m128 *squelch) {

    uint8_t buf8[MATRIX_WIDTH] __attribute__((aligned (16)));
    uint64_t depth;
    uint64_t len;
    uint64_t result[DEFAULT_BUF_SIZE];
    __m128 buf[DEFAULT_BUF_SIZE];

    while (!exitFlag) {
        len = 0;
        len += readFile(buf8, DEFAULT_BUF_SIZE, NULL, squelch, buf, inFile);

        if (!exitFlag && len) {
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
