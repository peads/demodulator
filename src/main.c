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
            "vpaddb all_nonetwentysevens(%%rip), %1, %0\n\t"
            "vpmovsxbw %0, %0\n\t"
            "vpmovsxwd %0, %0\n\t"
            "vcvtdq2ps %0, %0\n\t"
            "orq %2, %2\n\t"                        // if squelch != NULL
            "jz nosquelch\n\t"                      // apply squelch
            "vmulps %0, %0, %%xmm2\n\t"
            "vpermilps $0xB1, %%xmm2, %%xmm3\n\t"
            "vaddps %%xmm2, %%xmm3, %%xmm2\n\t"
            "vmulps all_hundredths(%%rip), %%xmm2, %%xmm2\n\t"
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
