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
#include "nvidia.cuh"

__global__
void fmDemod(const uint8_t *buf, const uint32_t len, float *result) {

    uint32_t i;
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t step = blockDim.x * gridDim.x;
    float ar, aj, br, bj, zr, zj;

    for (i = index; i < len; i += step) {

        ar = __int2float_rd(buf[i] + buf[i+2] - 254);//__hadd(buf[i] - 127, (buf[i + 2] - 127)));
        aj = __int2float_rd(254 - buf[i+1] - buf[i+3]);//-__hadd(buf[i + 1] - 127, (buf[i + 3] - 127)));

        br = __int2float_rd(buf[i+4] + buf[i+6] - 254);//__hadd(buf[i + 4] - 127, (buf[i + 6] - 127)));
        bj = __int2float_rd(buf[i+5] + buf[i+7] - 254);//__hadd(buf[i + 5] - 127, (buf[i + 7] - 127)));

        zr = __fmaf_rd(ar, br, -__fmul_rd(aj, bj));
        zj = __fmaf_rd(ar, bj, __fmul_rd(aj, br));

        result[i >> 2] = atan2f(zj, zr);
    }
}

static int8_t processMatrix(float squelch, FILE *inFile, struct chars *chars, FILE *outFile) {

    static const int grid_dim = (DEFAULT_BUF_SIZE + BLOCKDIM - 1) / BLOCKDIM;

    uint8_t *buf;
    float *result;

    int8_t exitFlag = 0;
    size_t readBytes;

    cudaMallocManaged(&buf, DEFAULT_BUF_SIZE*INPUT_ELEMENT_BYTES);
    cudaMallocManaged(&result, QTR_BUF_SIZE*OUTPUT_ELEMENT_BYTES);

    while (!exitFlag) {

        readBytes = fread(buf, INPUT_ELEMENT_BYTES, DEFAULT_BUF_SIZE, inFile);

        if (exitFlag = ferror(inFile)) {
            perror(nullptr);
            break;
        } else if (feof(inFile)) {
            exitFlag = EOF;
        }

        fmDemod<<<grid_dim, BLOCKDIM>>>(buf, readBytes, result);
        cudaDeviceSynchronize();

        fwrite(result, OUTPUT_ELEMENT_BYTES, QTR_BUF_SIZE, outFile);
    }

    cudaFree(buf);
    cudaFree(result);
    return exitFlag;
}

int main(int argc, char **argv) {

    int opt;
    float temp = 0.f;
    FILE *inFile = nullptr;
    FILE *outFile = nullptr;
    struct chars chars{};
    chars.isOt = 0;
    chars.isRdc = 0;

    if (argc < 3) {
        return -1;
    } else {
        while ((opt = getopt(argc, argv, "i:o:s:rf")) != -1) {
            switch (opt) {
                case 'r':
                    chars.isRdc = 1;
                    break;
                case 'f':
                    chars.isOt = 1;
                    break;
//                case 'd': // TODO reimplement downsmapling?
//                    chars.downsample = atoi(optarg);
//                    break;
                case 's':   // TODO add parameter to take into account the impedance of the system
                    // currently calculated for 50 Ohms (i.e. Prms = ((I^2 + Q^2)/2)/50 = (I^2 + Q^2)/100)
                    temp = exp10f((float) atof(optarg) / 10.f);
                    break;
                case 'i':
                    if (!strstr(optarg, "-")) {
                        inFile = fopen(optarg, "rb");
                    } else {
                        freopen(nullptr, "rb", stdin);
                        inFile = stdin;
                    }
                    break;
                case 'o':
                    if (!strstr(optarg, "-")) {
                        outFile = fopen(optarg, "wb");
                    } else {
                        freopen(nullptr, "wb", stdout);
                        outFile = stdout;
                    }
                    break;
                default:
                    break;
            }
        }
    }

    return processMatrix(temp, inFile, &chars, outFile) != EOF;
}