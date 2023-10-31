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
#include "matrix.h"
//#define BW
typedef __m256 (*butterWorthScalingFn_t)(__m256, __m256);

static inline __m256i shiftOrigin(__m256i u) {

    static const __m256i shift = {
            //_mm256_set1_epi8(-127);
            -0x7e7e7e7e7e7e7e7f,
            -0x7e7e7e7e7e7e7e7f,
            -0x7e7e7e7e7e7e7e7f,
            -0x7e7e7e7e7e7e7e7f
    };

    return _mm256_add_epi8(u, shift);
}

static inline void convert_epi8_ps(__m256i u, __m256 *uhi, __m256 *ulo, __m256 *vhi, __m256 *vlo) {

    __m256i temp[2];
    temp[0] = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(u));
    temp[1] = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(u, 1));
    *ulo = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(temp[0])));
    *uhi = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(temp[0], 1)));
    *vlo = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(temp[1])));
    *vhi = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(temp[1], 1)));
}

static inline __m256 scaleButterworthDcBlock(__attribute__((unused)) const __m256 wc, __m256 u) {

    return _mm256_rcp_ps(u);
}

//static inline __m256 scaleButterworthHighpass(__m256 wc, __m256 u) {
//
//    return _mm256_mul_ps(_mm256_rcp_ps(u), wc);
//}

/// The reciprocal of omega c must be passed in!
static inline __m256 scaleButterworthLowpass(const __m256 wc, __m256 u) {

    return _mm256_mul_ps(u, wc);
}

static inline __m256 filterButterWorth(__m256 u, const __m256 wc, const butterWorthScalingFn_t fn) {

    // Degree 8 coefficients
    static const __m256 BW_CONSTS[] = {
            {0.19509f,-0.980785f,0.19509f,-0.980785f,0.19509f,-0.980785f,0.19509f,-0.980785f},
            {0.55557f,-0.83147f,0.55557f,-0.83147f,0.55557f,-0.83147f,0.55557f,-0.83147f},
            {0.83147f,-0.55557f,0.83147f,-0.55557f,0.83147f,-0.55557f,0.83147f,-0.55557f},
            {0.980785f,-0.19509f,0.980785f,-0.19509f,0.980785f,-0.19509f,0.980785f,-0.19509f},
            {0.980785f,0.19509f,0.980785f,0.19509f,0.980785f,0.19509f,0.980785f,0.19509f},
            {0.83147f,0.55557f,0.83147f,0.55557f,0.83147f,0.55557f,0.83147f,0.55557f},
            {0.55557f,0.83147f,0.55557f,0.83147f,0.55557f,0.83147f,0.55557f,0.83147f},
            {0.19509f,0.980785f,0.19509f,0.980785f,0.19509f,0.980785f,0.19509f,0.980785f}};

    __m256 v = fn(wc, u);
    __m256 temp, acc = {1,1,1,1,1,1,1,1};
    size_t i;

    for (i = 0; i < 8; ++i) {
        temp = _mm256_add_ps(v, BW_CONSTS[i]);
        acc = _mm256_mul_ps(acc, temp);
    }
    v = _mm256_mul_ps(u, _mm256_rcp_ps(acc));

    return v;
}

#ifdef BW
static inline __m256 filterRealButterworth(__m256 u, const __m256 wc, const butterWorthScalingFn_t fn) {
    // Degree 16
    static const __m256 ONES = {1,1,1,1,1,1,1,1};
    static const __m256 BW_CONSTS[] = {
            {0.196034f, 0.196034f, 0.196034f, 0.196034f, 0.196034f, 0.196034f, 0.196034f, 0.196034f},
            {0.580569f, 0.580569f, 0.580569f, 0.580569f, 0.580569f, 0.580569f, 0.580569f, 0.580569f},
            {0.942793f, 0.942793f, 0.942793f, 0.942793f, 0.942793f, 0.942793f, 0.942793f, 0.942793f},
            {1.26879f,  1.26879f,  1.26879f,  1.26879f,  1.26879f,  1.26879f,  1.26879f,  1.26879f},
            {1.54602f,  1.54602f,  1.54602f,  1.54602f,  1.54602f,  1.54602f,  1.54602f,  1.54602f},
            {1.76384f,  1.76384f,  1.76384f,  1.76384f,  1.76384f,  1.76384f,  1.76384f,  1.76384f},
            {1.91388f,  1.91388f,  1.91388f,  1.91388f,  1.91388f,  1.91388f,  1.91388f,  1.91388f},
            {1.99037f,  1.99037f,  1.99037f,  1.99037f,  1.99037f,  1.99037f,  1.99037f,  1.99037f}};

    size_t i;
    __m256 acc = ONES, v = fn(wc, u),
        squared = _mm256_mul_ps(v, v);
    for (i = 0; i < 8; ++i) {
        acc = _mm256_mul_ps(acc, _mm256_add_ps(ONES,
                _mm256_add_ps(squared, _mm256_add_ps(v, BW_CONSTS[i]))));
    }
    return _mm256_mul_ps(u, _mm256_rcp_ps(acc));
}
#endif

static inline __m256 hComplexMulByConj(__m256 u) {

    static const __m256 indexComplexConjugate = {1, 1, -1.f, 1, 1, 1, -1.f, 1};
    static const __m256i indexOrdering = {
            // _mm256_setr_epi32(0,2,4,6,1,3,5,7);
            0x200000000,
            0x600000004,
            0x300000001,
            0x700000005};

    __m256 temp = _mm256_mul_ps(
            _mm256_permute_ps(u, _MM_SHUFFLE(1, 0, 1, 0)),              // abab
            _mm256_mul_ps(_mm256_permute_ps(u, _MM_SHUFFLE(2, 3, 3, 2)),// cd(-d)c
                    indexComplexConjugate));
    return _mm256_permutevar8x32_ps(_mm256_add_ps(temp,
            _mm256_permute_ps(temp, _MM_SHUFFLE(2, 3, 0, 1))), indexOrdering);
}

static inline __m256 hPolarDiscriminant_ps(__m256 u, __m256 v) {

    v = hComplexMulByConj(v);
    return _mm256_blend_ps(hComplexMulByConj(u), _mm256_permute2f128_ps(v, v, 1), 0b11110000);
}

static inline __m256 fmDemod(__m256 u) {

    static const __m256 all64s = {64.f, 64.f, 64.f, 64.f, 64.f, 64.f, 64.f, 64.f};
    static const __m256 all23s = {23.f, 23.f, 23.f, 23.f, 23.f, 23.f, 23.f, 23.f};
    static const __m256 all41s = {41.f, 41.f, 41.f, 41.f, 41.f, 41.f, 41.f, 41.f};
    static const __m256i index = {
            // _mm256_setr_epi32(1, 3, 5, 7, 0, 2, 3, 6);
            0x300000001,
            0x700000005,
            0x200000000,
            0x600000003
    };
    // fast atan2(y,x) := 64y/(23x+41*Sqrt[x^2+y^2])
    __m256 v = _mm256_mul_ps(u, u),
            hypot = _mm256_permute_ps(v, _MM_SHUFFLE(2, 3, 0, 1));
    hypot = _mm256_add_ps(v, hypot);
    hypot = _mm256_sqrt_ps(hypot);

    // 64*y
    v = _mm256_fmadd_ps(all23s, u, _mm256_mul_ps(all41s, hypot));
    // 1/(23*x+41*hypot)
    v = _mm256_permute_ps(_mm256_rcp_ps(v), _MM_SHUFFLE(2, 3, 0, 1));
    // 64*y/(23*x*41*hypot)
    u = _mm256_mul_ps(_mm256_mul_ps(all64s, u), v);

    // NAN check
    u = _mm256_and_ps(u, _mm256_cmp_ps(u, u, 0));
    u = _mm256_permutevar8x32_ps(u,index);
    return u;
}

void *processMatrix(void *ctx) {

    static const __m256 lowpassWc = {
            0.00008f,0.00008f,0.00008f,0.00008f,
            0.00008f,0.00008f,0.00008f,0.00008f};
    static const __m256 highpassWc = {
            1.f, 1.f, 1.f, 1.f,
            1.f, 1.f, 1.f, 1.f};
    consumerArgs *args = ctx;
    size_t i;
    __m256 result = {};
    __m256 hBuf[4] = {};
    __m256i u;
    uint8_t *buf = _mm_malloc(DEFAULT_BUF_SIZE, ALIGNMENT);
    while (!args->exitFlag) {
        sem_wait(args->full);
        pthread_mutex_lock(&args->mutex);
        memcpy(buf, args->buf, DEFAULT_BUF_SIZE);
        pthread_mutex_unlock(&args->mutex);
        sem_post(args->empty);

        for (i = 0; i < DEFAULT_BUF_SIZE; i += 32) {
            u = shiftOrigin(*(__m256i *) (buf + i));
            convert_epi8_ps(u, &hBuf[1],&hBuf[0],&hBuf[3],&hBuf[2]);

            hBuf[0] = filterButterWorth(hBuf[0], lowpassWc, scaleButterworthLowpass);
            hBuf[1] = filterButterWorth(hBuf[1], lowpassWc, scaleButterworthLowpass);
            hBuf[2] = filterButterWorth(hBuf[2], lowpassWc, scaleButterworthLowpass);
            hBuf[3] = filterButterWorth(hBuf[3], lowpassWc, scaleButterworthLowpass);

            hBuf[0] = filterButterWorth(hBuf[0], highpassWc, scaleButterworthDcBlock);
            hBuf[1] = filterButterWorth(hBuf[1], highpassWc, scaleButterworthDcBlock);
            hBuf[2] = filterButterWorth(hBuf[2], highpassWc, scaleButterworthDcBlock);
            hBuf[3] = filterButterWorth(hBuf[3], highpassWc, scaleButterworthDcBlock);

            hBuf[0] = hPolarDiscriminant_ps(hBuf[0], hBuf[1]);
            hBuf[0] = fmDemod(hBuf[0]);

            hBuf[2] = hPolarDiscriminant_ps(hBuf[2], hBuf[3]);
            hBuf[1] = fmDemod(hBuf[2]);

            result = _mm256_blend_ps(hBuf[0], _mm256_permute2f128_ps(hBuf[1], hBuf[1], 1), 0b11110000);
#ifdef BW
                result = filterRealButterworth(result, _mm256_set1_ps(1.f/12500), scaleButterworthLowpass);
#endif

            fwrite(&result, sizeof(__m256), 1, args->outFile);
        }
    }

    _mm_free(buf);
    return NULL;
}

void allocateBuffer(void **buf, const size_t len) {

    *buf = _mm_malloc(len, ALIGNMENT);
}
