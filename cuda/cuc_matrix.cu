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
#include <cuComplex.h>
#include "nvidia.cuh"

__global__
void fmDemod(uint8_t *idata, const uint32_t len, const float invert, const float gain, float *result) {

    uint32_t i;
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t step = blockDim.x * gridDim.x;
    cuComplex z;

    for (i = index; i < len; i += step + 2) {
        z = cuCmulf(
            {(float) (idata[i] + idata[i + 2] - 254), (float) (254 - idata[i + 1] - idata[i + 3])},
            {(float) (idata[i + 4] + idata[i + 6] - 254), (float) (idata[i + 5] + idata[i + 7] - 254)});

        z.y = __fmul_rn(64.f, z.y);
        z.x = __fmul_rn(z.y, __frcp_rn(__fmaf_rn(23.f, z.x, 41.f)));

        // for some reason it doesn't like copysignf in release builds
        z.x *= invert;
        result[i >> 2] = isnan(z.x) ? 0.f : gain != 0.f ? gain * z.x : z.x; //copysignf(z.x, invert) * gain : copysignf(z.x, invert);
    }
}

extern "C" int processMatrix(FILE *__restrict__ inFile,
                             const uint8_t mode,
                             float gain,
                             void *__restrict__ outFile) {

    int exitFlag = 0;
    uint8_t *dBuf;
    uint8_t *hBuf;
    float *dResult;
    float *hResult;
    size_t readBytes;

    const float invert = mode ? -1.f : 1.f;
    gain = gain != 1.f ? gain : 0.f;

    cudaMalloc(&dBuf, DEFAULT_BUF_SIZE);
    cudaMalloc(&dResult, (DEFAULT_BUF_SIZE >> 2) * sizeof(float));
    cudaMallocHost(&hBuf, DEFAULT_BUF_SIZE);
    cudaMallocHost(&hResult, (DEFAULT_BUF_SIZE >> 2) * sizeof(float));

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
        fmDemod<<<GRIDDIM, BLOCKDIM>>>(dBuf, DEFAULT_BUF_SIZE, invert, gain, dResult);
        cudaMemcpy(hResult,
            dResult,
            (DEFAULT_BUF_SIZE >> 2) * sizeof(float),
            cudaMemcpyDeviceToHost);

        fwrite(hResult, sizeof(float), (readBytes + 2) >> 2, (FILE *) outFile);
    }

    cudaFreeHost(hBuf);
    cudaFreeHost(hResult);
    cudaFree(dBuf);
    cudaFree(dResult);
    return exitFlag;
}
