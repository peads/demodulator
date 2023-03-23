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
 * as a--effectively--a packed float. TODO doesn't account for special cases,
 * but also doesn't seem to negatively affect demodulation, strangely. Should
 * probably start handling those two cases again, eventually.
 **/
extern uint64_t arg(__m128 a, __m128 b);
__asm__(
".section:\n\t"
".p2align 4\n\t"
"PI: "
    ".quad 0x40490fdb\n\t"
".text\n\t"

#ifdef __clang__
"_arg: "
#else
"arg: "
#endif
//    "vpxor %ymm3, %ymm3, %ymm3\n\t"         // store zero

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

    // "vcmpps $0xC, %ymm3, %ymm0, %ymm5\n\t" // TODO store mask more permanently
                                              // broadcast the salient value and
                                              // "and" it with the result later?

"gotime: "
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
    "vmovq %xmm0, %rax\n\t"
    "ret\n\t"
);

/**
 * Takes a 4x4 matrix and applies it to a 4x1 vector.
 * Here, it is used to apply the same rotation matrix to
 * two complex numbers. i.e., for the the matrix
 * T = {{a,b}, {c,d}} and two vectors {u1,u2} and {v1,v2}
 * concatenated, s.t. u = {u1,u2,v1,v2}, Tu =
 * {a*u1 + c*u1, b*u2 + d*u2, ... , b*v2 + d*v2}
 */
extern __m128 apply4x4_4x1Transform(const struct rotationMatrix *T, __m128 u);
__asm__(
#ifdef __clang__
"_apply4x4_4x1Transform: "
#else
"apply4x4_4x1Transform: "
#endif
    "vmulps 16(%rdi), %xmm0, %xmm2\n\t"      // u1*a11, u2*a12, u3*a13, ...
    "vmulps (%rdi), %xmm0, %xmm1\n\t"        // u1*a21, u2*a22, ...
    "vpermilps $0xB1, %xmm2, %xmm0\n\t"
    "vaddps %xmm2, %xmm0, %xmm2\n\t"         // u1*a11 + u2*a12, ... , u3*a13 + u4*a14
    "vpermilps $0xB1, %xmm1, %xmm0\n\t"
    "vaddps %xmm1, %xmm0, %xmm1\n\t"         // u1*a21 + u2*a22, ... , u3*a23 + u4*a24
    "vblendps $0xA, %xmm2, %xmm1, %xmm0\n\t" // u1*a11 + u2*a12, u1*a21 + u2*a22,
    "ret"                                    // u3*a13 + u4*a14, u3*a23 + u4*a24
);

static inline struct rotationMatrix generateRotationMatrix(const float theta, const float phi) {

    const float cosT = cosf(theta);
    const float sinP = sinf(phi);

    struct rotationMatrix result = {
        .a1 = {cosT, -sinP, cosT, -sinP},
        .a2 = {sinP, cosT, sinP, cosT}
    };

    return result;
}

extern uint64_t filter(__m128 *buf, uint64_t len, uint8_t downsample);
__asm__(
#ifdef __clang__
"_filter: "
#else
"filter: "
#endif
    "movq %rdx, %r8\n\t"
    "negq %r8\n\t"
"L1: "
    "xorq %rax, %rax\n\t"
"L2: "
    "movq %rax, %rcx\n\t"
    "shlq $4, %rcx\n\t"
    "vpermilps $0x4E, (%rdi, %rcx), %xmm0\n\t"
    "vaddps (%rdi, %rcx), %xmm0, %xmm0\n\t"
    "shrq $5, %rcx\n\t"
    "shlq $4, %rcx\n\t"
    "vmovaps %xmm0, (%rdi, %rcx)\n\t"
    "addq $1, %rax\n\t"
    "cmp %rsi, %rax\n\t"
    "jl L2\n\t"
    "addq $1, %r8\n\t"
    "jl L1\n\t"

    "movq %rdx, %rcx\n\t"
    "movq %rsi, %rax\n\t"
    "shr %cl, %rax\n\t"
    "ret"
);

static void removeDCSpike(__m128 *buf, const uint32_t len) {

    static __m128 dcAvgIq = {0, 0, 0, 0};
    uint64_t i;

    for (i = 0; i < len; i += 16) {

        PERFORM_DESPIKE(dcAvgIq, buf, i)
        PERFORM_DESPIKE(dcAvgIq, buf, i + 1)
        PERFORM_DESPIKE(dcAvgIq, buf, i + 2)
        PERFORM_DESPIKE(dcAvgIq, buf, i + 3)
        PERFORM_DESPIKE(dcAvgIq, buf, i + 4)
        PERFORM_DESPIKE(dcAvgIq, buf, i + 5)
        PERFORM_DESPIKE(dcAvgIq, buf, i + 6)
        PERFORM_DESPIKE(dcAvgIq, buf, i + 7)
        PERFORM_DESPIKE(dcAvgIq, buf, i + 8)
        PERFORM_DESPIKE(dcAvgIq, buf, i + 9)
        PERFORM_DESPIKE(dcAvgIq, buf, i + 10)
        PERFORM_DESPIKE(dcAvgIq, buf, i + 11)
        PERFORM_DESPIKE(dcAvgIq, buf, i + 12)
        PERFORM_DESPIKE(dcAvgIq, buf, i + 13)
        PERFORM_DESPIKE(dcAvgIq, buf, i + 14)
        PERFORM_DESPIKE(dcAvgIq, buf, i + 15)
    }
}

static void rotateForNonOffsetTuning(__m128 *buf, const uint32_t len) {

    uint64_t i;

    for (i = 0; i < len; i += 16) {
        buf[i] = apply4x4_4x1Transform(&CONJ_TRANSFORM, buf[i]);
        buf[i + 1] = apply4x4_4x1Transform(&CONJ_TRANSFORM, buf[i + 1]);
        buf[i + 2] = apply4x4_4x1Transform(&CONJ_TRANSFORM, buf[i + 2]);
        buf[i + 3] = apply4x4_4x1Transform(&CONJ_TRANSFORM, buf[i + 3]);
        buf[i + 4] = apply4x4_4x1Transform(&CONJ_TRANSFORM, buf[i + 4]);
        buf[i + 5] = apply4x4_4x1Transform(&CONJ_TRANSFORM, buf[i + 5]);
        buf[i + 6] = apply4x4_4x1Transform(&CONJ_TRANSFORM, buf[i + 6]);
        buf[i + 7] = apply4x4_4x1Transform(&CONJ_TRANSFORM, buf[i + 7]);
        buf[i + 8] = apply4x4_4x1Transform(&CONJ_TRANSFORM, buf[i + 8]);
        buf[i + 9] = apply4x4_4x1Transform(&CONJ_TRANSFORM, buf[i + 9]);
        buf[i + 10] = apply4x4_4x1Transform(&CONJ_TRANSFORM, buf[i + 10]);
        buf[i + 11] = apply4x4_4x1Transform(&CONJ_TRANSFORM, buf[i + 11]);
        buf[i + 12] = apply4x4_4x1Transform(&CONJ_TRANSFORM, buf[i + 12]);
        buf[i + 13] = apply4x4_4x1Transform(&CONJ_TRANSFORM, buf[i + 13]);
        buf[i + 14] = apply4x4_4x1Transform(&CONJ_TRANSFORM, buf[i + 14]);
        buf[i + 15] = apply4x4_4x1Transform(&CONJ_TRANSFORM, buf[i + 15]);
    }
}
static uint64_t demodulateFmData(__m128 *buf, const uint64_t len, float *result) {

    uint64_t i, j, ret;

    for (i = 0, j = 0; i < len; ++i, j += 2) {

        ret = arg(buf[i], buf[i+1]);
        memcpy(result + j, &ret, OUTPUT_ELEMENT_BYTES << 1);
    }

    return j;
}

static void checkFileStatus(FILE *file) {

    if (ferror(file)) {
        char errorMsg[256];
        sprintf(errorMsg, "\nI/O error when reading file");
        perror(errorMsg);
        exitFlag = 1;
    } else if (feof(file)) {
        fprintf(stderr, "\nExiting\n");
        exitFlag = EOF;
    }
}

static void *processMatrix(struct readArgs *args) {
    
    uint64_t depth;
    float *result;

    if (args->isRdc) {
        removeDCSpike(args->buf, args->len);
    }

    if (!args->isOt) {
        rotateForNonOffsetTuning(args->buf, args->len);
    }

    depth = filter(args->buf, args->len, args->downsample);
    result = calloc(depth << 1, OUTPUT_ELEMENT_BYTES);
    depth = demodulateFmData(args->buf, depth, result);


    fwrite(result, OUTPUT_ELEMENT_BYTES, depth, args->outFile);
    free(result);

    return NULL;
}

static int readFileData(struct readArgs *args) {

    union {
        uint8_t buf[MATRIX_WIDTH];
        __m128i v;
    } z;

    uint64_t j = 0;
    FILE *inFile = args->inFile ? fopen(args->inFile, "rb") : stdin;

    args->len = DEFAULT_BUF_SIZE;
    args->buf = calloc(args->len, MATRIX_ELEMENT_BYTES);
    args->outFile = args->outFileName ? fopen(args->outFileName, "wb") : stdout;

    while (!exitFlag) {

        fread(z.buf, INPUT_ELEMENT_BYTES, MATRIX_WIDTH, inFile);
        checkFileStatus(inFile);

        __asm__ (
            "vpaddb _Z(%%rip), %1, %0\n\t"
            "vpmovsxbw %0, %0\n\t"
            "vpmovsxwd %0, %0\n\t"
            "vcvtdq2ps %0, %0\n\t"
            "orq %2, %2\n\t"
            "jz nosquelch\n\t"
            "vmulps %0, %0, %%xmm2\n\t"
            "vpermilps $0xB1, %%xmm2, %%xmm3\n\t"
            "vaddps %%xmm2, %%xmm3, %%xmm2\n\t"
            "vmulps _ALL_HUNDREDTHS(%%rip), %%xmm2, %%xmm2\n\t"
            "vcmpps $0x1D, (%2), %%xmm2, %%xmm2\n\t"
            "vandps %%xmm2, %0, %0\n\t"
        "nosquelch: "
        :"=x"(args->buf[j++]):"x"(z.v),"r"(args->squelch):"xmm2","xmm3");

        if (!exitFlag && j >= args->len) {
            args->len = j;
            processMatrix(args);
            j = 0;
        }
    }

    fclose(inFile);
    fclose(args->outFile);
    free(args->buf);
    if (args->squelch) {
        free(args->squelch);
    }

    return exitFlag;
}

int main(int argc, char **argv) {

    static struct readArgs args;
    int opt;
    int exitCode;

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
                    args.squelch = malloc(MATRIX_ELEMENT_BYTES);
                    *args.squelch = _mm_set1_ps(powf(10.f, atof(optarg) / 10.f));
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

    exitCode = readFileData(&args) != EOF ? 1 : 0;
    fprintf(stderr, "%s\n", exitCode ? "Exited with error" : "Exited");
    return exitCode;
}
