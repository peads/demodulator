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
void fmDemod(uint8_t *idata, const uint32_t len, const float squelchLevel, float *result) {

    uint32_t i;
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t step = blockDim.x * gridDim.x;
    cuComplex z;

    for (i = index; i < len; i += step + 8) {

        // power = (I^2 + Q^2) / 2 / (50 Ohms) = (I^2 + Q^2) / (100 Ohms)
        // power = (I^2 + Q^2) / 2 / (50 Ohms) = (I^2 + Q^2) / (100 Ohms)
        if (squelchLevel != 0.f && (float)(
                (idata[i] * idata[i]) + (idata[i + 1] * idata[i + 1]) +
                (idata[i + 2] * idata[i + 2]) + (idata[i + 3] * idata[i + 3]) +
                (idata[i + 4] * idata[i + 4]) + (idata[i + 5] * idata[i + 5]) +
                (idata[i + 6] * idata[i + 4]) + (idata[i + 7] * idata[i + 7]))
            * 0.01f < squelchLevel) {
            result[i >> 2] = 0.f;
        } else {

            z = cuCmulf(
                {(float)(idata[i] + idata[i + 2] - 254), (float)(254 - idata[i + 1] - idata[i + 3])},
                {(float)(idata[i + 4] + idata[i + 6] - 254), (float)(idata[i + 5] + idata[i + 7] - 254)});

            z.y = __fmul_rn(64.f, z.y);
            z.x = __fmul_rn(z.y, __frcp_rn(__fmaf_rn(23.f, z.x, 41.f)));
            result[i >> 2] = isnan(z.x) ? 0.f : z.x; // delay line
//        result[i >> 2] = atan2f(z.y, z.x);
        }
    }
}

extern "C" int processMatrix(FILE *__restrict__ inFile,
                             const uint8_t mode,
                             float gain,
                             void *__restrict__ outFile) {

    int exitFlag = mode != 1;
    uint8_t *dBuf;
    uint8_t *hBuf;
    float *dResult;
    float *hResult;
    size_t readBytes;
    float squelchLevel = 750.f;
//    const uint8_t isGain = fabsf(1.f - gain) > GAIN_THRESHOLD;

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
        cudaDeviceSynchronize();
        fmDemod<<<GRIDDIM, BLOCKDIM>>>(dBuf, DEFAULT_BUF_SIZE, squelchLevel, dResult);
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
