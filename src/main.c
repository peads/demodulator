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
 * Takes packed float representing the complex numbers
 * (ar + iaj), (br + ibj), s.t. z = {ar, aj, br, bj}
 * and returns their argument as a float
 **/
extern float arg(__m128 z);
__asm__(

".section:\n\t"
    ".p2align 4\n\t"
"LC0: "
    ".quad 4791830004637892608\n\t"
"LC1: "
    ".quad 4735535009282654208\n\t"
"LC2: "
    ".quad 4765934306774482944\n\t"
"LC3: "
    ".quad 0x40490fdb\n\t"
"LC4: "
    ".quad 4333543705419175690\n\t"
".text\n\t"

#ifdef __clang__
"_arg: "
#else
"arg: "
#endif
    "vpxor %xmm3, %xmm3, %xmm3\n\t"         // store zero
    "vpermilps $0xEB, %xmm0, %xmm1\n\t"     // (ar, aj, br, bj) => (aj, aj, ar, ar)
    "vpermilps $0x5, %xmm0, %xmm0\n\t"      // and                 (bj, br, br, bj)

    "vmulps %xmm1, %xmm0, %xmm0\n\t"        // aj*bj, aj*br, ar*br, ar*bj
    "vpermilps $0x8D, %xmm0, %xmm2\n\t"     // aj*br, aj*bj, ar*bj, ar*br
    "vaddsubps %xmm2, %xmm0, %xmm0\n\t"     //  ... [don't care], ar*bj + aj*br, ar*br - aj*bj, [don't care] ...
    "vmulps %xmm0, %xmm0, %xmm1\n\t"        // ... , (ar*bj + aj*br)^2, (ar*br - aj*bj)^2, ...
    "vpermilps $0x1B, %xmm1, %xmm2\n\t"
    "vaddps %xmm2, %xmm1, %xmm1\n\t"        // ..., (ar*br - aj*bj)^2 + (ar*bj + aj*br)^2, ...

    "vpermilps $0x01, %xmm0, %xmm2\n\t"
    "vcomiss %xmm2, %xmm3\n\t"
    "jnz showtime\n\t"
    "vpermilps $0x02, %xmm0, %xmm2\n\t"
    "vcomiss %xmm3, %xmm2\n\t"
    "jz zero\n\t"
    "ja showtime\n\t"
    "vmovq LC3(%rip), %xmm0\n\t"
    "ret \n\t"

"zero: "
    "vmovq %xmm3,%xmm0\n\t"
    "ret\n\t"

"showtime: "                                // approximating atan2 with atan(z)
                                            //   = z/(1 + (9/32) z^2) for z = (64 y)/(23 x + 41 Sqrt[x^2 + y^2])
    "vrsqrtps %xmm1, %xmm1\n\t"             // ..., 1/Sqrt[(ar*br - aj*bj)^2 + (ar*bj + aj*br)^2], ...
    "vmulps %xmm1, %xmm0, %xmm0\n\t"        // ... , zj/||z|| , zr/||z|| = (ar*br - aj*bj) / Sqrt[(ar*br - aj*bj)^2 + (ar*bj + aj*br)^2], ...
    "movddup LC0(%rip), %xmm2\n\t"          // 64
    "movddup LC1(%rip), %xmm3\n\t"          // 23

    "vmulps %xmm2, %xmm0, %xmm2\n\t"        // 64*zj
    "vmulps %xmm3, %xmm0, %xmm3\n\t"        // 23*zr
    "movddup LC2(%rip), %xmm0\n\t"          // 41
    "vaddps %xmm3, %xmm0, %xmm3\n\t"        // 23*zr + 41
    "vpermilps $0x1B, %xmm3, %xmm3\n\t"
    "vrcpps %xmm3, %xmm3\n\t"
    "vmulps %xmm3, %xmm2, %xmm0\n\t"

    "vpermilps $0x01, %xmm0, %xmm0\n\t"
    "ret\n\t"
);

extern __m128 applySquelch(__m128 z, __m128 squelch);
__asm__(
#ifdef __clang__
"_applySquelch: "
#else
"applySquelch: "
#endif
    "vmulps %xmm0, %xmm0, %xmm2\n\t"
    "vpermilps $0xB1, %xmm2, %xmm3\n\t"
    "vaddps %xmm2, %xmm3, %xmm2\n\t"
    "vmovddup LC4(%rip), %xmm3\n\t"
    "vmulps %xmm3, %xmm2, %xmm2\n\t"
    "vcmpps $0x1D, %xmm1, %xmm2, %xmm2\n\t"
    "vandps %xmm2, %xmm0, %xmm0\n\t"
    "ret"
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

static uint64_t downSample(__m128 *buf, uint32_t len, const uint8_t downsample) {

    uint64_t i, j;

    for (j = 0; j < downsample; ++j) {
        for (i = 0; i < len; ++i) {
            buf[i >> 1] = _mm_add_ps(buf[i], _mm_permute_ps(buf[i],
                    _MM_SHUFFLE(1, 0, 3, 2)));
        }
    }

    return len >> downsample;
}

static void removeDCSpike(__m128 *buf, const uint32_t len) {

    static __m128 dcAvgIq = {0, 0, 0, 0};
    uint64_t i;

    for (i = 0; i < len; ++i) {
        dcAvgIq = _mm_add_ps(dcAvgIq, _mm_mul_ps(DC_RAW_CONST, _mm_sub_ps(buf[i], dcAvgIq)));
        buf[i] = _mm_sub_ps(buf[i], dcAvgIq);
    }
}

static void rotateForNonOffsetTuning(__m128 *buf, const uint32_t len) {

    uint64_t i;

    for (i = 0; i < len; ++i) {
        buf[i] = apply4x4_4x1Transform(&CONJ_TRANSFORM, buf[i]);
    }
}

static uint64_t demodulateFmData(__m128 *buf, const uint32_t len, float **result) {

    uint64_t i, j;

    *result = calloc(len << 1, OUTPUT_ELEMENT_BYTES);
    for (i = 0, j = 0; i < len; ++i, j += 2) {
        (*result)[j] = arg(_mm_mul_ps(buf[i], NEGATE_B_IM));
        (*result)[j + 1] = arg(_mm_mul_ps(_mm_blend_ps(buf[i], buf[i + 1], 0b0011), NEGATE_B_IM));
    }

    return j;
}

static void checkFileStatus(FILE *file) {

    if (ferror(file)) {
        char errorMsg[256];
        sprintf(errorMsg, "I/O error when reading file");
        perror(errorMsg);
        exitFlag = 1;
    } else if (feof(file)) {
        fprintf(stderr, "Exiting\n");
        exitFlag = EOF;
    }
}

static void *processMatrix(void *ctx) {

    struct readArgs *args = ctx;
    uint64_t depth;
    float *result;

    if (args->isRdc) {
        removeDCSpike(args->buf, args->len);
    }

    if (!args->isOt) {
        rotateForNonOffsetTuning(args->buf, args->len);
    }

    depth = downSample(args->buf, args->len, args->downsample);
    depth = demodulateFmData(args->buf, depth, &result);


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

        //1000039be: 0f 6f 04 24                 	movq	(%rsp), %mm0
        //1000039c2: 0f f8 44 24 08              	psubb	8(%rsp), %mm0
        //1000039c7: 0f ef c9                    	pxor	%mm1, %mm1
        //1000039ca: 0f 64 c8                    	pcmpgtb	%mm0, %mm1
        //1000039cd: 0f 60 c1                    	punpcklbw	%mm1, %mm0
        //1000039d0: 0f ef c9                    	pxor	%mm1, %mm1
        //1000039d3: 0f 65 c8                    	pcmpgtw	%mm0, %mm1
        //1000039d6: 0f 6f d0                    	movq	%mm0, %mm2
        //1000039d9: 0f 69 d1                    	punpckhwd	%mm1, %mm2
        //1000039dc: c5 f8 57 c0                 	vxorps	%xmm0, %xmm0, %xmm0
        //1000039e0: 0f 2a c2                    	cvtpi2ps	%mm2, %xmm0
        //1000039e3: c5 fb 12 c0                 	vmovddup	%xmm0, %xmm0
        //1000039e7: 0f 61 c1                    	punpcklwd	%mm1, %mm0
        //1000039ea: 0f 2a c0                    	cvtpi2ps	%mm0, %xmm0
        /* all THAT for ` args->buf[j] = _mm_cvtpi8_ps(_mm_sub_pi8(z.v, Z));`? good God clang, wtf? */

        __asm__ (
            "vpaddb _Z(%%rip), %1, %0\n\t"
            "vpmovsxbw %0, %0\n\t"
            "vpmovsxwd %0, %0\n\t"
            "vcvtdq2ps %0, %0\n\t"
        :"=x"(args->buf[j]):"x"(z.v));

        if (args->squelch) {
            args->buf[j] = applySquelch(args->buf[j], *args->squelch);
        }
        j++;

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

    if (argc < 3) {
        return -1;
    } else {
        while ((opt = getopt(argc, argv, "r:i:o:d:f:s:")) != -1) {
            switch (opt) {
                case 'r':
                    args.isRdc = atoi(optarg);
                    break;
                case 'f':
                    args.isOt = atoi(optarg);
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
                    if (NULL == strstr(optarg, "-")) {
                        args.inFile = optarg;
                    } else {
                        freopen(NULL, "rb", stdin);
                    }
                    break;
                case 'o':
                    if (NULL == strstr(optarg, "-")) {
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
    return readFileData(&args) != EOF ? 1 : 0;
}
