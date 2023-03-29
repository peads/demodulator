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
                        //rdi           rsi         xmm0            rdx             rcx            r8                  r9
extern uint64_t readFile(uint8_t *buf, uint64_t len, __m128 squelch, __m128 *result, FILE *inFile, struct chars *chars, FILE *outFile);
__asm__(
#ifdef __clang__
"_readFile: "
#else
"readFile: "
#endif
    "pushq %rbp\n\t"
    "movq %rsp, %rbp\n\t"
//    "addq $-8, %rsp\n\t"
    "pushq %rbx\n\t"
    "pushq %r12\n\t"
    "pushq %r13\n\t"
    "pushq %r14\n\t"
    "pushq %r15\n\t"
    "pushq %rdx\n\t"
    "pushq %r9\n\t"
    "pushq (%r8)\n\t"

    "shlq $4, %rsi\n\t"
    "addq %rsi, %rdx\n\t"   // store address of end of array
    "negq %rsi\n\t"

    "leaq (%rcx), %r15\n\t" // file
    "movq %rdx, %r14\n\t"   // result
    "movq %rsi, %r12\n\t"   // len
    "movq %rdi, %rbx\n\t"   // buf
"L7: "
    "xorq %r13, %r13\n\t"       // ret = 0
"L6: "
    "movq %rbx, %rdi\n\t"    // set the fread arguments, args->file
    "movq $1, %rsi\n\t"
    "movq $4, %rdx\n\t"
    "movq %r15, %rcx\n\t"
    "call _fread\n\t"               // TODO change macro-ifdef-defined label
    "addq %rax, %r13\n\t"

    "movq %r15, %rdi\n\t"
    "callq _checkFileStatus\n\t"    // TODO consider inlining file err/eof check
                                    // TODO change macro-ifdef-defined label

    "vmovaps (%rbx), %xmm1\n\t" // args->buf
    "vpaddb all_nonetwentysevens(%rip), %xmm1, %xmm1\n\t"
    "vpmovsxbw %xmm1, %xmm1\n\t"
    "vpmovsxwd %xmm1, %xmm1\n\t"
    "vcvtdq2ps %xmm1, %xmm1\n\t"
    "vmovq %xmm0, %rcx\n\t"   // args->squelch
    "test %rcx, %rcx\n\t"        // if squelch != NULL
    "jz nosquelch\n\t"          // apply squelch
    "vmulps %xmm1, %xmm1, %xmm0\n\t"
    "vpermilps $0xB1, %xmm0, %xmm3\n\t"
    "vaddps %xmm0, %xmm3, %xmm0\n\t"
    "vmulps all_hundredths(%rip), %xmm0, %xmm0\n\t"
    "vcmpps $0x1D, %xmm1, %xmm0, %xmm0\n\t"
    "vandps %xmm0, %xmm1, %xmm1\n\t"
"nosquelch:\n\t"
    "vmovaps %xmm1, (%r12, %r14)\n\t"
    "addq $16, %r12\n\t"
    "jl L6\n\t"

    "movq _exitFlag(%rip), %rcx\n\t" // TODO put exitFlag in header.S
    "test %rcx, %rcx\n\t"
    "jnz end\n\t"

    "test %r13, %r13\n\t"
    "jz loopEnd\n\t"

    "movb (%rsp), %cl\n\t"
    "testb %cl, %cl\n\t"
    "jz isNotRdc\n\t"           // if isRdc
// removeDCSpike(buf128, DEFAULT_BUF_SIZE);
    "movq %r14, %rdi\n\t"
    "movq $1024, %rsi\n\t"
    "callq _removeDCSpike\n\t" // TODO change to actual label name once in own .S file
"isNotRdc: "
    "movb 1(%rsp), %cl\n\t"
    "testb %cl, %cl\n\t"
    "jnz isOt\n\t"           // if !isOt
// applyComplexConjugate(buf128, DEFAULT_BUF_SIZE);
    "movq %r14, %rdi\n\t"
    "movq $1024, %rsi\n\t"
    "callq _applyComplexConjugate\n\t" // TODO change to actual label name once in own .S file
"isOt: "
// filter(buf128, DEFAULT_BUF_SIZE, downsample);
    "movq %r14, %rdi\n\t"
    "movq $1024, %rsi\n\t"
    "movb 2(%rsp), %dl\n\t"
    "callq _filter\n\t" // TODO change to actual label
// demodulateFmData(buf128, depth, result);
    "movq %r14, %rdi\n\t"
    "movq %rax, %rsi\n\t"
    "movq _result(%rip), %rdx\n\t"
    "callq _demodulateFmData\n\t" // TODO change to actual label
// fwrite(result, OUTPUT_ELEMENT_BYTES, depth, outFile);
    "movq _result(%rip), %rdi\n\t" // TODO result put in header.S
    "movq $1024, %rsi\n\t"
    "movq %rax, %rdx\n\t"
    "movq (%rsp), %rdi\n\t"
    "callq _fwrite\n\t" // TODO change macro-ifdef-defined label
"loopEnd: "
    "jmp L7\n\t"

"end: "
    "popq %r9\n\t"  // 0 - outFile
    "popq %r8\n\t"  // 16 - chars
    "popq %rdx\n\t" // 32 - buf128
    "popq %r15\n\t"
    "popq %r14\n\t"
    "popq %r13\n\t"
    "popq %r12\n\t"
    "popq %rbx\n\t"
//    "addq $8, %rsp\n\t"
    "popq %rbp\n\t"
//    "movq %r13, %rax\n\t"
    "ret"
);

void processMatrix(FILE *inFile, FILE *outFile, uint8_t downsample, uint8_t isRdc, uint8_t isOt, __m128 squelch) {

//    uint64_t depth = 0;
//    uint64_t ret = 0;
    uint8_t buf[MATRIX_WIDTH] __attribute__((aligned (16)));
    __m128 buf128[DEFAULT_BUF_SIZE];
    struct chars chars;

    chars.isRdc = isRdc;
    chars.isOt = isOt;
    chars.downsample = downsample;

    readFile(buf, DEFAULT_BUF_SIZE, squelch, buf128, inFile, &chars, outFile);
//    while (1) {
//
//        ret = readFile(buf, DEFAULT_BUF_SIZE, squelch, buf128, inFile, &chars, outFile);
//
//        if (exitFlag) {
//            break;
//        }
//
//        if (ret) {
//            if (isRdc) {
//                removeDCSpike(buf128, DEFAULT_BUF_SIZE);
//            }
//
//            if (!isOt) {
//                applyComplexConjugate(buf128, DEFAULT_BUF_SIZE);
//            }
//
//            depth = filter(buf128, DEFAULT_BUF_SIZE, downsample);
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
