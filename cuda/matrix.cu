/*
 * This file is part of the demodulator distribution
 * (https://github.com/peads/demodulator).
 * with code originally part of the misc_snippets distribution
 * (https://github.com/peads/misc_snippets).
 * Copyright (c) 2023-2024 Patrick Eads.
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
#include "matrix.cuh"

__global__
static void correctIq(
        const float esr,
        const uint8_t *__restrict__ in,
        const size_t len,
        float *__restrict__ out) {

    size_t i;
    static float off[2] = {};

    for (i = 0; i < len >> 1; i += 2) {
        out[i] = ((float) in[i]) - off[0];
        out[len - i - 2] = ((float) in[len - i - 2]) - off[0];

        out[i+1] = ((float) in[i + 1]) - off[1];
        out[len - i - 1] = ((float) in[len - i - 1]) - off[1];

        off[0] += (out[i] + out[len - i - 2]) * esr;
        off[1] += (out[i+1] + out[len - i - 1]) * esr;
    }
}

__global__
static void fmDemod(const float *in, const size_t len, float *out) {

    size_t i;
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t step = blockDim.x * gridDim.x;
    float zr, zj;

    for (i = index; i < len; i += step) {

        // ac-b(-d)=ac+bd
        // a(-d)+bc=-ad+bc
        zr = in[i] * in[i + 2] + in[i + 1] * in[i + 3];
        zj = -in[i] * in[i + 3] + in[i + 1] * in[i + 2];

        out[i >> 2] = atan2(zj, zr);
    }
}

extern "C" void *processMatrix(void *ctx) {

    auto *args = static_cast<consumerArgs *>(ctx);
    float *dResult;
    float *dfBuf;
    uint8_t *dBuf;
    float *hResult;

    cudaMalloc(&dBuf, DEFAULT_BUF_SIZE);
    cudaMalloc(&dfBuf, DEFAULT_BUF_SIZE * sizeof(float));
    cudaMalloc(&dResult, (DEFAULT_BUF_SIZE >> 2) * sizeof(float));
    cudaMallocHost(&hResult, (DEFAULT_BUF_SIZE >> 2) * sizeof(float));

    auto esr = (float) (50. / args->sampleRate);

    while (!args->exitFlag) {

        sem_wait(args->full);
        pthread_mutex_lock(&args->mutex);
        cudaMemcpy(dBuf, args->buf, DEFAULT_BUF_SIZE, cudaMemcpyHostToDevice);
        pthread_mutex_unlock(&args->mutex);
        sem_post(args->empty);

        cudaDeviceSynchronize();
        correctIq<<<GRIDDIM, BLOCKDIM>>>(esr, dBuf, DEFAULT_BUF_SIZE, dfBuf);
        fmDemod<<<GRIDDIM, BLOCKDIM>>>(dfBuf, DEFAULT_BUF_SIZE, dResult);
        cudaDeviceSynchronize();
        cudaMemcpy(hResult, dResult, (DEFAULT_BUF_SIZE >> 2) * sizeof(float),
                cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        fwrite(hResult, sizeof(float), DEFAULT_BUF_SIZE >> 2, args->outFile);
    }

    cudaFreeHost(args->buf);
    cudaFreeHost(hResult);
    cudaFree(dfBuf);
    cudaFree(dBuf);
    cudaFree(dResult);

    return nullptr;
}

extern "C" void allocateBuffer(void **buf, const size_t len) {

    cudaMallocHost(buf, len);
}
