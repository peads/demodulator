/*
 * This file is part of the demodulator distribution
 * (https://github.com/peads/demodulator).
 * and code originally part of the misc_snippets distribution
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

static uint8_t isRdc;
static uint8_t isOffsetTuning;

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

static inline __m128 mm256Epi8convertmmPs(__m256i data) {

    __m128i lo_lane = _mm256_castsi256_si128(data);
    return _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_cvtepi8_epi16(lo_lane)));
}

/**
 * Takes a 4x4 matrix and applies it to a 4x1 vector.
 * Here, it is used to apply the same rotation matrix to
 * two complex numbers. i.e., for the the matrix
 * T = {{a,b}, {c,d}} and two vectors {u1,u2} and {v1,v2}
 * concatenated, s.t. u = {u1,u2,v1,v2}, Tu =
 * {a*u1 + c*u1, b*u2 + d*u2, ... , b*v2 + d*v2}
 */
static __m128 apply4x4_4x1Transform(const struct rotationMatrix T, const __m128 u) {

    __m128 temp, temp1;

    temp = _mm_mul_ps(T.a1.v, u);           // u1*a11, u2*a12, u3*a13, ...
    temp1 = _mm_mul_ps(T.a2.v, u);          // u1*a21, u2*a22, ...
    return _mm_blend_ps(_mm_add_ps(temp,       // u1*a11 + u2*a12, ... , u3*a13 + u4*a14
                                   _mm_permute_ps(temp, _MM_SHUFFLE(2,3,0,1))),
                        _mm_add_ps(temp1,              // u1*a21 + u2*a22, ... , u3*a23 + u4*a24
                                   _mm_permute_ps(temp1, _MM_SHUFFLE(2,3,0,1))),
                        0xA);                                     // u1*a11 + u2*a12, u1*a21 + u2*a22,
    // u3*a13 + u4*a14, u3*a23 + u4*a24
    // A = 0000 1010 = 00 22 => _MM_SHUFFLE(0,0,2,2)
}

static inline struct rotationMatrix generateRotationMatrix(const float theta, const float phi) {

    const float cosT = cosf(theta);
    const float sinT = sinf(phi);

    struct rotationMatrix result = {
            .a1 = {cosT, -sinT, cosT, -sinT},
            .a2 = {sinT, cosT, sinT, cosT}
    };

    return result;
}

static uint64_t downSample(__m128 *buf, uint32_t len, const uint8_t downsample) {

    uint64_t i,j;

    for (j = 0; j < downsample; ++j) {
        for (i = 0; i < len; ++i) {
            buf[i >> 1] = _mm_add_ps(buf[i],_mm_permute_ps(buf[i],
                                                           _MM_SHUFFLE(1, 0, 3, 2)));
        }
    }

    return len >> downsample;
}

static void removeDCSpike(__m128 *buf, const uint32_t len) {

    static const __m128 ratio = {1e-05f,1e-05f,1e-05f,1e-05f};
    static __m128 dcAvgIq = {0,0,0,0};

    uint64_t i;

    for (i = 0; i < len; ++i) {
        dcAvgIq = _mm_add_ps(dcAvgIq, _mm_mul_ps(ratio, _mm_sub_ps(buf[i], dcAvgIq)));
        buf[i] = _mm_sub_ps(buf[i], dcAvgIq);
    }
}

static void rotateForNonOffsetTuning(__m128 *buf, const uint32_t len) {

    uint64_t i;

    for(i = 0; i < len; ++i) {
        buf[i] = apply4x4_4x1Transform(CONJ_TRANSFORM, buf[i]);
    }
}

static uint64_t demodulateFmData(__m128 *buf, const uint32_t len, float **result) {

    uint64_t i, j;

    *result = calloc(len << 1, OUTPUT_ELEMENT_BYTES);
    for (i = 0, j = 0; i < len; ++i, j += 2) {
        (*result)[j] = arg(_mm_mul_ps( buf[i], NEGATE_B_IM));
        (*result)[j+1] = arg(_mm_mul_ps(_mm_blend_ps(buf[i], buf[i+1], 0b0011), NEGATE_B_IM));
    }

    return j;
}

static uint64_t splitIntoRows(const uint8_t *buf, const uint64_t len, __m128 *result, __m128 *squelch) {

    uint64_t j = 0;
    uint64_t i;
    int64_t leftToProcess = len;
    __m128 rms, mask;

    union {
        uint8_t buf[MATRIX_WIDTH];
        __m256i v;
    } z;

    for (i = 0; leftToProcess > 0; i += MATRIX_WIDTH) {

        memcpy(z.buf, buf + i, MATRIX_WIDTH);
        result[j] = mm256Epi8convertmmPs(_mm256_sub_epi8(z.v, Z));

        if (squelch) {
            rms = _mm_mul_ps(result[j], result[j]);
            rms = _mm_mul_ps(HUNDREDTH,
                             _mm_add_ps(rms, _mm_permute_ps(rms, _MM_SHUFFLE(2, 3, 0, 1))));
            mask = _mm_cmp_ps(rms, *squelch, _CMP_GE_OQ);
            result[j] = _mm_and_ps(result[j], mask);
        }
        j++;
        leftToProcess -= MATRIX_WIDTH;
    }

    return j;
}

//static uint64_t processMatrix(__m128 *buf, const uint64_t len) {
//
////    uint64_t depth;
////    uint64_t count = len & 3UL // len/MATRIX_WIDTH + (len % MATRIX_WIDTH != 0 ? 1 : 0))
////                     ? (len >> LOG2_VECTOR_WIDTH) + 1UL
////                     : (len >> LOG2_VECTOR_WIDTH);
//
////    *buff = calloc(count << 2, MATRIX_ELEMENT_BYTES);
//
////    depth = splitIntoRows(buf, len, *buff, squelch);
//
//
//
//    return len;
//}

static void handleFileError(FILE *file) {

    char errorMsg[256];
    int fileStatus = ferror(file) | feof(file);

    switch (fileStatus) {
        case 0:
            break;
        case 1:
        default:
            sprintf(errorMsg, "I/O error when reading file");
            perror(errorMsg);
        case EOF:
            exitFlag = 1;
            break;
    }

    exitFlag = fileStatus;
}
struct readArgs {
    char *inFile;
    char *outFileName;
    uint8_t downsample;
    uint8_t isRdc;
    uint8_t isOt;
    __m128 *squelch;
    __m128 *buf;
    uint64_t len;
    FILE *outFile;
};

static void *runReadStreamData(void *ctx) { // TODO pass struct of cmd line args

    struct readArgs *args = ctx;
    uint64_t depth;
//    __m128 *lowPassed;
    float *result;

    if (args->isRdc) {
        removeDCSpike(args->buf, args->len);
    }

    if (!args->isOt) {
        rotateForNonOffsetTuning(args->buf, args->len);
    }

    depth = downSample(args->buf, args->len, args->downsample);
    depth = demodulateFmData(args->buf, depth, &result);

    fprintf(stderr, "writing out!");

    fwrite(result, OUTPUT_ELEMENT_BYTES, depth, args->outFile);
    free(result);

    return NULL;
}

static inline int readFileData(struct readArgs *args) {

    union {
        uint8_t buf[MATRIX_WIDTH];
        __m256i v;
    } z;

    uint64_t j = 0;
    pthread_t pid;

    FILE *inFile = args->inFile ? fopen(args->inFile, "rb") : stdin;

    args->len = DEFAULT_BUF_SIZE >> 2;
    args->buf = calloc(args->len, MATRIX_ELEMENT_BYTES);
    args->outFile = args->outFileName ? fopen(args->outFileName, "wb") : stdout;


    while (!exitFlag) {

        fread(z.buf, INPUT_ELEMENT_BYTES, MATRIX_WIDTH, inFile);
        handleFileError(inFile);

        args->buf[j++] = mm256Epi8convertmmPs(_mm256_sub_epi8(z.v, Z));

        if (j >= args->len) {
            pthread_create(&pid, NULL, runReadStreamData, args);
            j = 0;
        }
    }

    pthread_join(pid, NULL);
    fclose(inFile);
    fclose(args->outFile);
    free(args->buf);

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
#ifndef DEBUG
                case 'i':
                    if (NULL == strstr(optarg, "-")) {
                        args.inFile = optarg;
                    } else {
//                        freopen(NULL, "rb", stdin);
                    }
                    break;
                case 'o':
                    if (NULL == strstr(optarg, "-")) {
                        args.outFileName = optarg;
                    } else {
//                        freopen(NULL, "wb", stdout);
                    }
                    break;
#endif
                default:
                    break;
            }
        }
    }
    return readFileData(&args);
}
