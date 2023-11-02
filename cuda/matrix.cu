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
    float a, b, c, d, ac, bd, zr, zj;

    for (i = index; i < len; i += step) {

        a = 0.25f * (float) (buf[i] + buf[i + 2] + buf[i + 4] + buf[i + 6] - 508);          // ar
        b = 0.25f * (float) (buf[i + 1] + buf[i + 3] + buf[i + 5] + buf[i + 7] - 508);      // aj
        c = 0.25f * (float) (buf[i + 8] + buf[i + 10] + buf[i + 12] + buf[i + 14] - 508);   // br
        d = 0.25f * (float) (508 - buf[i + 9] - buf[i + 11] - buf[i + 13] - buf[i + 15]);   // -bj

        ac = a * c;
        bd = b * d;
        zr = ac - bd;
        zj = (a + b) * (c + d) - (ac + bd);

        // ||z|| = Sqrt[zr*zr+zj*zj], fatan2f(y,x) = 64*y/(41 + 23*x),
        // arg(z) = fatan2(zj/||z||, zr/||z||)
        // = 64*(zj/Sqrt[zr*zr+zj*zj]) / (41 + 23*(zr/Sqrt[zr*zr+zj*zj]))
        // = 64*zj / (||z|| * (41 + 23*(zr/||z||)))
        // = 64*zj / (41*||z|| + 23*zr)
//        zr = 64.f * zj * __frcp_rn(23.f * zr + 41.f * hypotf(zr, zj));
//        result[i >> 3] = isnan(zr) ? 0.f : zr;
        result[i >> 3] = atan2(zj, zr);
    }
}

extern "C" void *processMatrix(void *ctx) {

    auto *args = static_cast<consumerArgs *>(ctx);
    float *dResult;
    uint8_t *dBuf;
    float *hResult;

    cudaMalloc(&dBuf, DEFAULT_BUF_SIZE);
    cudaMalloc(&dResult, (DEFAULT_BUF_SIZE >> 3) * sizeof(float));
    cudaMallocHost(&hResult, (DEFAULT_BUF_SIZE >> 3) * sizeof(float));

    while (!args->exitFlag) {

        sem_wait(args->full);
        pthread_mutex_lock(&args->mutex);
        cudaMemcpy(dBuf, args->buf, DEFAULT_BUF_SIZE, cudaMemcpyHostToDevice);
        pthread_mutex_unlock(&args->mutex);
        sem_post(args->empty);

        cudaDeviceSynchronize();
        fmDemod<<<GRIDDIM, BLOCKDIM>>>(dBuf, DEFAULT_BUF_SIZE, dResult);

        cudaDeviceSynchronize();
        cudaMemcpy(hResult,
                dResult,
                (DEFAULT_BUF_SIZE >> 3) * sizeof(float),
                cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();
        fwrite(hResult, sizeof(float), DEFAULT_BUF_SIZE >> 3, args->outFile);
    }

    cudaFreeHost(args->buf);
    cudaFreeHost(hResult);
    cudaFree(dBuf);
    cudaFree(dResult);

    return nullptr;
}

extern "C" void allocateBuffer(void **buf, const size_t len) {

    cudaMallocHost(buf, len);
}
