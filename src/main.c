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
//__attribute__((used)) extern size_t	 fread(void * __restrict ptr, size_t size, size_t nitems, FILE * __restrict stream);
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
}                   //rdi           rsi         rdx         rcx             r8              r9
extern uint64_t foo(uint8_t *buf, uint64_t len, __m128 *u, __m128 *squelch, __m128 *result, FILE *file);
//    len += fread(buf, INPUT_ELEMENT_BYTES, MATRIX_WIDTH, file);
__asm__(
#ifdef __clang__
"_foo: "
#else
"foo: "
#endif
    "pushq %rbp\n\t"
    "movq %rsp, %rbp\n\t"

    "xorq %r11, %r11\n\t"
    "shlq $4, %rsi\n\t"
    "addq %rsi, %r8\n\t"
    "negq %rsi\n\t"
"L6: "
//    "leaq (%rdi), %rdi\n\t"
    "pushq %rsi\n\t"    // preserve our registers
    "pushq %rdx\n\t"
    "pushq %rcx\n\t"
    "pushq %rdi\n\t"
    "pushq %r8\n\t"
    "pushq %r9\n\t"
    "movq $1, %rsi\n\t" // set the fread arguments (rdi is implicit)
    "movq $4, %rdx\n\t"
    "leaq (%r9), %rcx\n\t"
//    "addq $-8, %rsp\n\t"
    "call _fread\n\t"
//    "addq $8, %rsp\n\t"
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
//    "addq $-8, %rsp\n\t"
    "callq _checkFileStatus\n\t"
//    "addq $8, %rsp\n\t"
    "popq %r9\n\t"
    "popq %r8\n\t"
    "popq %rdi\n\t"
    "popq %rcx\n\t"
    "popq %rdx\n\t"
    "popq %rsi\n\t"

    "vmovaps (%rdx), %xmm1\n\t"
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

static __m128 buf[DEFAULT_BUF_SIZE];

void processMatrix(FILE *inFile, FILE *outFile, uint8_t downsample, uint8_t isRdc, uint8_t isOt, __m128 *squelch) {

    static union {
        uint8_t buf[MATRIX_WIDTH];  // TODO see if I can get this to just be byte-aligned array to
                                    // get rid of the union altogether
        __m128 u;
    } z;
    uint64_t depth;
    uint64_t len;
    uint64_t result[DEFAULT_BUF_SIZE];

    while (!exitFlag) {
        len = 0;
        len += foo(z.buf, DEFAULT_BUF_SIZE, &z.u, squelch, buf, inFile);
//        for(j = 0, len = 0; j < DEFAULT_BUF_SIZE; ++j) {
//            __asm__ (
//                    "xorq %rsi, %rsi\n\t"
//                    "xorq %%r10, %%r10\n\t"
//                    "movq $1024, %rsi\n\t"
//                    "shlq $4, %rsi\n\t"
//                    "movq %2, %%r8\n\t"
//                    "addq %rsi, %%r8\n\t"
//                    "negq %rsi\n\t"
//                "L6: "
//                    "leaq (%6), %%rdi\n\t"
//                    "movq $1, %%rsi\n\t"
//                    "movq $4, %%rdx\n\t"
//                    "leaq (%5), %%rcx\n\t"
//                    "pushq %%rcx\n\t"
//                    "addq $-8, %%rsp\n\t"
//                    "call _fread\n\t"
//                    "addq %%rax, %1\n\t"
//                    "addq $8, %%rsp\n\t"
//                    "popq %%rdi\n\t"
//                    "callq _checkFileStatus\n\t"
//
//                    "vmovaps (%%r8, %rsi), %%xmm4\n\t"
//                    "vpaddb all_nonetwentysevens(%%rip), %3, %%xmm4\n\t"
//                    "vpmovsxbw %%xmm4, %%xmm4\n\t"
//                    "vpmovsxwd %%xmm4, %%xmm4\n\t"
//                    "vcvtdq2ps %%xmm4, %%xmm4\n\t"
//                    "orq %4, %4\n\t"                    // if squelch != NULL
//                    "jz nosquelch\n\t"                  // apply squelch
//                    "vmulps %%xmm4, %%xmm4, %%xmm2\n\t"
//                    "vpermilps $0xB1, %%xmm2, %%xmm3\n\t"
//                    "vaddps %%xmm2, %%xmm3, %%xmm2\n\t"
//                    "vmulps all_hundredths(%%rip), %%xmm2, %%xmm2\n\t"
//                    "vcmpps $0x1D, (%4), %%xmm2, %%xmm2\n\t"
//                    "vandps %%xmm2, %%xmm4, %%xmm4\n\t"
//                "nosquelch:\n\t"
//                    "add $16, %rsi\n\t"
//                    "jl L6\n\t"
//                   "" : "+r"(j), "=r"(len)
//                    : "r"(buf), "x"(z.u), "r"(squelch), "r"(inFile), "r"(z.buf)
//                    : "rdi", "rsi", "rdx", "rcx", "xmm2", "xmm3", "xmm4", "r8", "r9");
//        }

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
