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
void fmDemod(const uint8_t *buf, const uint32_t len, const float gain, float *result) {

    uint32_t i;
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t step = blockDim.x * gridDim.x;
    float a, b, c, d, ac, bd, zr, zj;

    for (i = index; i < len; i += step) {

        a = __int2float_rn(buf[i] + buf[i + 2] - 254);
        b = __int2float_rn(254 - buf[i + 1] - buf[i + 3]);

        c = __int2float_rn(buf[i + 4] + buf[i + 6] - 254);
        d = __int2float_rn(buf[i + 5] + buf[i + 7] - 254);

        ac = a * c;
        bd = b * d;
        zr = ac - bd;
        zj = __fmaf_rn((a + b), (c + d), -(ac + bd));

        zj = __fmul_rn(64.f, zj);
        zr = __fmul_rn(zj, __frcp_rn(__fmaf_rn(23.f, zr, 41.f)));

        result[i >> 2] = isnan(zr) ? 0.f : gain ? gain * zr : zr; // delay line
    }
}

extern "C" int processMatrix(FILE *__restrict__ inFile,
                             const uint8_t mode,
                             float gain,
                             void *__restrict__ outFile) {

    int exitFlag = 0;
    uint8_t *dBuf;
    float *dResult;
    uint8_t *hBuf;
    float *hResult;
    size_t readBytes;
    gain = gain != 1.f ? gain : 0.f;

    cudaMallocHost(&hBuf, DEFAULT_BUF_SIZE);
    cudaMalloc(&dBuf, DEFAULT_BUF_SIZE);
    cudaMallocHost(&hResult, (DEFAULT_BUF_SIZE >> 2) * OUTPUT_ELEMENT_BYTES);
    cudaMalloc(&dResult, (DEFAULT_BUF_SIZE >> 2) * OUTPUT_ELEMENT_BYTES);

    hBuf[0] = 0;
    hBuf[1] = 0;

    while (!exitFlag) {

        readBytes = fread(hBuf + 2, 1, DEFAULT_BUF_SIZE - 2, inFile);

        if ((exitFlag = ferror(inFile))) {
            perror(nullptr);
            break;
        } else if (feof(inFile)) {
            exitFlag = EOF;
        }

        cudaMemcpy(dBuf, hBuf, DEFAULT_BUF_SIZE, cudaMemcpyHostToDevice);

        fmDemod<<<GRIDDIM, BLOCKDIM>>>(dBuf, readBytes + 2, gain, dResult);

        cudaMemcpy(hResult,
                dResult,
                (DEFAULT_BUF_SIZE >> 2) * OUTPUT_ELEMENT_BYTES,
                cudaMemcpyDeviceToHost);

        fwrite(hResult, OUTPUT_ELEMENT_BYTES, DEFAULT_BUF_SIZE >> 2, (FILE *) outFile);
    }

    cudaFreeHost(hBuf);
    cudaFreeHost(hResult);
    cudaFree(dBuf);
    cudaFree(dResult);
    return exitFlag;
}
