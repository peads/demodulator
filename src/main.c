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

/**
 * Takes packed floats representing two sets of complex numbers
 * of the form (ar + iaj), (br + ibj), s.t. z = {ar, aj, br, bj}
 * and returns the arguments of the phasors (i.e. Arg[(ar + iaj).(br + ibj)])
 * as a--effectively--a packed float. Doesn't account for special
 * case x<0 && y==0, but this doesn't seem to negatively affect performance.
 **/
__asm__(
#ifdef __clang__
"_arg: "
#else
"arg: "
#endif
    "vblendps $0b0011, %xmm1, %xmm0, %xmm1\n\t"
    "vinsertf128 $1, %xmm1, %ymm0, %ymm0\n\t"
    "vmulps _NEGATE_B_IM(%rip), %ymm0, %ymm0\n\t" // (ar, aj, br, -bj)

    "vpermilps $0xEB, %ymm0, %ymm1\n\t"     // (ar, aj, br, bj) => (aj, aj, ar, ar)
    "vpermilps $0x5, %ymm0, %ymm0\n\t"      // and                 (bj, br, br, bj)

    "vmulps %ymm1, %ymm0, %ymm0\n\t"        // aj*bj, aj*br, ar*br, ar*bj
    "vpermilps $0x8D, %ymm0, %ymm2\n\t"     // aj*br, aj*bj, ar*bj, ar*br
    "vaddsubps %ymm2, %ymm0, %ymm0\n\t"     //  ... [don't care], ar*bj + aj*br, ar*br - aj*bj, [don't care] ...
    "vmulps %ymm0, %ymm0, %ymm1\n\t"        // ... , (ar*bj + aj*br)^2, (ar*br - aj*bj)^2, ...
    "vpermilps $0x1B, %ymm1, %ymm2\n\t"
    "vaddps %ymm2, %ymm1, %ymm1\n\t"        // ..., (ar*br - aj*bj)^2 + (ar*bj + aj*br)^2, ...

    "vrsqrtps %ymm1, %ymm1\n\t"             // ..., 1/Sqrt[(ar*br - aj*bj)^2 + (ar*bj + aj*br)^2], ...
    "vmulps %ymm1, %ymm0, %ymm0\n\t"        // ... , zj/||z|| , zr/||z|| = (ar*br - aj*bj) / Sqrt[(ar*br - aj*bj)^2 + (ar*bj + aj*br)^2], ...

    "vmulps _ALL_64S(%rip), %ymm0, %ymm2\n\t"        // 64*zj
    "vmulps _ALL_23S(%rip), %ymm0, %ymm3\n\t"        // 23*zr
    "vaddps _ALL_41S(%rip), %ymm3, %ymm3\n\t"        // 23*zr + 41
    "vpermilps $0x1B, %ymm3, %ymm3\n\t"
    "vrcpps %ymm3, %ymm3\n\t"
    "vmulps %ymm3, %ymm2, %ymm0\n\t"

    "vextractf128 $1, %ymm0, %xmm1\n\t"
    "vpermilps $1, %ymm0, %ymm0\n\t"
    "vblendps $1, %xmm0, %xmm1, %xmm0\n\t"
    "vcmpps $0x0, %xmm0, %xmm0, %xmm1\n\t"  // effectively the NAN check
    "vandps %xmm1, %xmm0, %xmm0\n\t"
    "vmovq %xmm0, %rax\n\t"
    "jmpq *%rdx\n\t"
);

extern uint64_t filter(__m128 *buf, uint64_t len, uint8_t downsample);
__asm__(
#ifdef __clang__
"_filter: "
#else
"filter: "
#endif
    "shlq $4, %rsi\n\t"
    "movq %rdx, %r8\n\t"
    "negq %r8\n\t"
"L1: "
    "xorq %rax, %rax\n\t"
"L2: "
    "movq %rax, %rcx\n\t"
//    "shlq $4, %rcx\n\t"
    "vpermilps $0x4E, (%rdi, %rcx), %xmm0\n\t"
    "vaddps (%rdi, %rcx), %xmm0, %xmm0\n\t"
    "shrq $5, %rcx\n\t"
    "shlq $4, %rcx\n\t"
    "vmovaps %xmm0, (%rdi, %rcx)\n\t"
    "addq $16, %rax\n\t"
    "cmp %rsi, %rax\n\t"
    "jl L2\n\t"
    "addq $1, %r8\n\t"
    "jl L1\n\t"

    "movq %rdx, %rcx\n\t"
    "addq $4, %rcx\n\t" // n >> 4; n >> downsample; == n >> (4 + downsample);
    "movq %rsi, %rax\n\t"
    "shr %cl, %rax\n\t"
    "ret"
);

extern void removeDCSpike(__m128 *buf, uint64_t len);
__asm__(
".data\n\t"
//".align 4\n\t"
"dc_avg_iq: .zero 16\n\t"
".text\n\t"
#ifdef __clang__
"_removeDCSpike: "
#else
"removeDCSpike: "
#endif
    "movq %rdi, %rcx\n\t"   // store array address
    "movq %rsi, %rax\n\t"   // store n
    "shlq $4, %rax\n\t"
    "addq %rax, %rcx\n\t"   // store address of end of array
    "negq %rax\n\t"
    "vmovaps dc_avg_iq(%rip), %xmm1\n\t"
"L3: "
    "vmovaps (%rcx,%rax), %xmm0\n\t"
    "vsubps %xmm1, %xmm0, %xmm1\n\t"
    "vmulps _DC_RAW_CONST(%rip), %xmm1, %xmm1\n\t"
    "vaddps %xmm1, %xmm1, %xmm1\n\t"
    "vsubps %xmm1, %xmm0, %xmm0\n\t"
    "vmovaps %xmm0, (%rcx,%rax)\n\t"
    // loop unroll one
    "vmovaps 16(%rcx,%rax), %xmm0\n\t"
    "vsubps %xmm1, %xmm0, %xmm1\n\t"
    "vmulps _DC_RAW_CONST(%rip), %xmm1, %xmm1\n\t"
    "vaddps %xmm1, %xmm1, %xmm1\n\t"
    "vsubps %xmm1, %xmm0, %xmm0\n\t"
    "vmovaps %xmm0, 16(%rcx,%rax)\n\t"
    // loop unroll two
    "vmovaps 32(%rcx,%rax), %xmm0\n\t"
    "vsubps %xmm1, %xmm0, %xmm1\n\t"
    "vmulps _DC_RAW_CONST(%rip), %xmm1, %xmm1\n\t"
    "vaddps %xmm1, %xmm1, %xmm1\n\t"
    "vsubps %xmm1, %xmm0, %xmm0\n\t"
    "vmovaps %xmm0, 32(%rcx,%rax)\n\t"
    // loop unroll three
    "vmovaps 48(%rcx,%rax), %xmm0\n\t"
    "vsubps %xmm1, %xmm0, %xmm1\n\t"
    "vmulps _DC_RAW_CONST(%rip), %xmm1, %xmm1\n\t"
    "vaddps %xmm1, %xmm1, %xmm1\n\t"
    "vsubps %xmm1, %xmm0, %xmm0\n\t"
    "vmovaps %xmm0, 48(%rcx,%rax)\n\t"
    // loop unroll four
    "vmovaps 64(%rcx,%rax), %xmm0\n\t"
    "vsubps %xmm1, %xmm0, %xmm1\n\t"
    "vmulps _DC_RAW_CONST(%rip), %xmm1, %xmm1\n\t"
    "vaddps %xmm1, %xmm1, %xmm1\n\t"
    "vsubps %xmm1, %xmm0, %xmm0\n\t"
    "vmovaps %xmm0, 64(%rcx,%rax)\n\t"
    // loop unroll five
    "vmovaps 80(%rcx,%rax), %xmm0\n\t"
    "vsubps %xmm1, %xmm0, %xmm1\n\t"
    "vmulps _DC_RAW_CONST(%rip), %xmm1, %xmm1\n\t"
    "vaddps %xmm1, %xmm1, %xmm1\n\t"
    "vsubps %xmm1, %xmm0, %xmm0\n\t"
    "vmovaps %xmm0, 80(%rcx,%rax)\n\t"
    // loop unroll six
    "vmovaps 96(%rcx,%rax), %xmm0\n\t"
    "vsubps %xmm1, %xmm0, %xmm1\n\t"
    "vmulps _DC_RAW_CONST(%rip), %xmm1, %xmm1\n\t"
    "vaddps %xmm1, %xmm1, %xmm1\n\t"
    "vsubps %xmm1, %xmm0, %xmm0\n\t"
    "vmovaps %xmm0, 96(%rcx,%rax)\n\t"
    // loop unroll seven
    "vmovaps 112(%rcx,%rax), %xmm0\n\t"
    "vsubps %xmm1, %xmm0, %xmm1\n\t"
    "vmulps _DC_RAW_CONST(%rip), %xmm1, %xmm1\n\t"
    "vaddps %xmm1, %xmm1, %xmm1\n\t"
    "vsubps %xmm1, %xmm0, %xmm0\n\t"
    "vmovaps %xmm0, 112(%rcx,%rax)\n\t"
    // i += 8
    "addq $128, %rax\n\t"
    "jl L3\n\t"
    "vmovaps %xmm1, dc_avg_iq(%rip)\n\t"
    "ret"
);

extern void applyComplexConjugate(__m128 *buf, uint64_t len);
__asm__(
#ifdef __clang__
"_applyComplexConjugate: "
#else
"applyComplexConjugate: "
#endif
    "movq %rdi, %rcx\n\t"   // store array address
    "movq %rsi, %rax\n\t"    // store n
    "shlq $4, %rax\n\t"
    "addq %rax, %rcx\n\t"    // store address of end of array
    "negq %rax\n\t"
"L5: "
    "vmovaps (%rcx,%rax), %xmm0\n\t"
    "vmulps _CNJ_TRANSFORM(%rip), %xmm0, %xmm0\n\t"
    "vmovaps %xmm0, (%rcx,%rax)\n\t"
    // loop unroll one
    "vmovaps 16(%rcx,%rax), %xmm0\n\t"
    "vmulps _CNJ_TRANSFORM(%rip), %xmm0, %xmm0\n\t"
    "vmovaps %xmm0, 16(%rcx,%rax)\n\t"
    // loop unroll two
    "vmovaps 32(%rcx,%rax), %xmm0\n\t"
    "vmulps _CNJ_TRANSFORM(%rip), %xmm0, %xmm0\n\t"
    "vmovaps %xmm0, 32(%rcx,%rax)\n\t"
    // loop unroll three
    "vmovaps 48(%rcx,%rax), %xmm0\n\t"
    "vmulps _CNJ_TRANSFORM(%rip), %xmm0, %xmm0\n\t"
    "vmovaps %xmm0, 48(%rcx,%rax)\n\t"
    // loop unroll four
    "vmovaps 64(%rcx,%rax), %xmm0\n\t"
    "vmulps _CNJ_TRANSFORM(%rip), %xmm0, %xmm0\n\t"
    "vmovaps %xmm0, 64(%rcx,%rax)\n\t"
    // loop unroll five
    "vmovaps 80(%rcx,%rax), %xmm0\n\t"
    "vmulps _CNJ_TRANSFORM(%rip), %xmm0, %xmm0\n\t"
    "vmovaps %xmm0, 80(%rcx,%rax)\n\t"
    // loop unroll six
    "vmovaps 96(%rcx,%rax), %xmm0\n\t"
    "vmulps _CNJ_TRANSFORM(%rip), %xmm0, %xmm0\n\t"
    "vmovaps %xmm0, 96(%rcx,%rax)\n\t"
    // loop unroll seven
    "vmovaps 112(%rcx,%rax), %xmm0\n\t"
    "vmulps _CNJ_TRANSFORM(%rip), %xmm0, %xmm0\n\t"
    "vmovaps %xmm0, 112(%rcx,%rax)\n\t"
    // i += 8
    "addq $128, %rax\n\t"
    "jl L5\n\t"
    "ret"
);

extern uint64_t demodulateFmData(__m128 *buf, uint64_t len, uint64_t *result);
__asm__(
#ifdef __clang__
"_demodulateFmData: "
#else
"demodulateFmData: "
#endif
    "movq %rdi, %rcx\n\t"   // store buf address
    "movq %rsi, %r8\n\t"    // store n
    "shlq $4, %r8\n\t"
    "addq %r8, %rcx\n\t"    // store address of end of buf

    "movq %rdx, %r9\n\t"    // store result address
    "shrq %r8\n\t"
    "movq %r8, %r10\n\t"
    "shlq %r8\n\t"
    "addq %r10, %r9\n\t"     // store address of end of result

    "negq %r8\n\t"
    "negq %r10\n\t"
"L4: "
    "vmovaps (%rcx,%r8), %xmm1\n\t"
    "vmovaps -16(%rcx,%r8), %xmm0\n\t"
    "leaq (%rip), %rdx\n\t"
    "addq $9, %rdx\n\t"
    "jmp _arg\n\t"              // TODO consider inlining arg?
    "movq %rax, (%r9,%r10)\n\t"
    // loop unroll one
    "vmovaps 16(%rcx,%r8), %xmm1\n\t"
    "vmovaps (%rcx,%r8), %xmm0\n\t"
    "leaq (%rip), %rdx\n\t"
    "addq $9, %rdx\n\t"
    "jmp _arg\n\t"
    "movq %rax, 8(%r9,%r10)\n\t"
    // loop unroll two
    "vmovaps 32(%rcx,%r8), %xmm1\n\t"
    "vmovaps 16(%rcx,%r8), %xmm0\n\t"
    "leaq (%rip), %rdx\n\t"
    "addq $9, %rdx\n\t"
    "jmp _arg\n\t"
    "movq %rax, 16(%r9,%r10)\n\t"
    // loop unroll three
    "vmovaps 48(%rcx,%r8), %xmm1\n\t"
    "vmovaps 32(%rcx,%r8), %xmm0\n\t"
    "leaq (%rip), %rdx\n\t"
    "addq $9, %rdx\n\t"
    "jmp _arg\n\t"
    "movq %rax, 24(%r9,%r10)\n\t"
    // loop unroll four
    "vmovaps 64(%rcx,%r8), %xmm1\n\t"
    "vmovaps 48(%rcx,%r8), %xmm0\n\t"
    "leaq (%rip), %rdx\n\t"
    "addq $9, %rdx\n\t"
    "jmp _arg\n\t"
    "movq %rax, 32(%r9,%r10)\n\t"
    // loop unroll five
    "vmovaps 80(%rcx,%r8), %xmm1\n\t"
    "vmovaps 64(%rcx,%r8), %xmm0\n\t"
    "leaq (%rip), %rdx\n\t"
    "addq $9, %rdx\n\t"
    "jmp _arg\n\t"
    "movq %rax, 40(%r9,%r10)\n\t"
    // loop unroll six
    "vmovaps 96(%rcx,%r8), %xmm1\n\t"
    "vmovaps 80(%rcx,%r8), %xmm0\n\t"
    "leaq (%rip), %rdx\n\t"
    "addq $9, %rdx\n\t"
    "jmp _arg\n\t"
    "movq %rax, 48(%r9,%r10)\n\t"
    // loop unroll seven
    "vmovaps 112(%rcx,%r8), %xmm1\n\t"
    "vmovaps 96(%rcx,%r8), %xmm0\n\t"
    "leaq (%rip), %rdx\n\t"
    "addq $9, %rdx\n\t"
    "jmp _arg\n\t"
    "movq %rax, 56(%r9,%r10)\n\t"
    // ++i, j += 2
    "addq $64, %r10\n\t"
    "addq $128, %r8\n\t"
    "jl L4\n\t"
    "shlq $1, %rsi\n\t"
    "movq %rsi, %rax\n\t"
    "ret"
);

static void checkFileStatus(FILE *file) {

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

static void processMatrix(struct readArgs *args) {
    
    uint64_t depth;

    if (args->isRdc) {
        removeDCSpike(args->buf, args->len);
    }

    if (!args->isOt) {
        applyComplexConjugate(args->buf, args->len);
    }

    depth = filter(args->buf, args->len, args->downsample);
    depth = demodulateFmData(args->buf, depth, args->result);


    fwrite(args->result, OUTPUT_ELEMENT_BYTES, depth, args->outFile);
}

static int readFileData(struct readArgs *args) {

    union {
        uint8_t buf[MATRIX_WIDTH];
        __m128i v;
    } z;

    uint64_t j = 0;
    FILE *inFile = args->inFile ? fopen(args->inFile, "rb") : stdin;

    args->len = DEFAULT_BUF_SIZE;
    args->buf = calloc(DEFAULT_BUF_SIZE, MATRIX_ELEMENT_BYTES);
    args->outFile = args->outFileName ? fopen(args->outFileName, "wb") : stdout;
    args->result = calloc(DEFAULT_BUF_SIZE >> (args->downsample - 1), OUTPUT_ELEMENT_BYTES);

    while (!exitFlag) {

        fread(z.buf, INPUT_ELEMENT_BYTES, MATRIX_WIDTH, inFile);
        checkFileStatus(inFile);

        __asm__ (
            "vpaddb _Z(%%rip), %1, %0\n\t"
            "vpmovsxbw %0, %0\n\t"
            "vpmovsxwd %0, %0\n\t"
            "vcvtdq2ps %0, %0\n\t"
            "orq %2, %2\n\t"                        // if squelch != NULL
            "jz nosquelch\n\t"                      // apply squelch
            "vmulps %0, %0, %%xmm2\n\t"
            "vpermilps $0xB1, %%xmm2, %%xmm3\n\t"
            "vaddps %%xmm2, %%xmm3, %%xmm2\n\t"
            "vmulps _ALL_HUNDREDTHS(%%rip), %%xmm2, %%xmm2\n\t"
            "vcmpps $0x1D, (%2), %%xmm2, %%xmm2\n\t"
            "vandps %%xmm2, %0, %0\n\t"
        "nosquelch: "
        :"=x"(args->buf[j++]):"x"(z.v),"r"(args->squelch):"xmm2","xmm3");

        if (!exitFlag && j >= DEFAULT_BUF_SIZE) {
            processMatrix(args);
            j = 0;
        }
    }

    fclose(inFile);
    fclose(args->outFile);
    free(args->buf);
    free(args->result);

    return exitFlag;
}

int main(int argc, char **argv) {

    static struct readArgs args;
    int opt;
    __m128 squelch;

    if (argc < 3) {
        return -1;
    } else {
        while ((opt = getopt(argc, argv, "i:o:d:s:rf")) != -1) {
            switch (opt) {
                case 'r':
                    args.isRdc = 1;
                    break;
                case 'f':
                    args.isOt = 1;
                    break;
                case 'd':
                    args.downsample = atoi(optarg);
                    break;
                case 's':   // TODO add parameter to take into account the impedance of the system
                            // currently calculated for 50 Ohms (i.e. Prms = ((I^2 + Q^2)/2)/50 = (I^2 + Q^2)/100)
                    squelch = _mm_set1_ps(powf(10.f, (float) atof(optarg) / 10.f));
                    args.squelch = &squelch;
                    break;
                case 'i':
                    if (!strstr(optarg, "-")) {
                        args.inFile = optarg;
                    } else {
                        freopen(NULL, "rb", stdin);
                    }
                    break;
                case 'o':
                    if (!strstr(optarg, "-")) {
                        args.outFileName = optarg;
                    } else {
                        freopen(NULL, "wb", stdout);
                    }
                    break;
                default:
                    break;
            }
        }
    }
#ifdef DEBUG
    int exitCode = readFileData(&args) != EOF;
    fprintf(stderr, "%s\n", exitCode ? "Exited with error" : "Exited");
    return exitCode;
#else
    return readFileData(&args) != EOF;
#endif
}
