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
#include <arm_neon.h>

#ifdef __ARM_FEATURE_SVE

#include <arm_sve.h>

#else
    #error "NO SVE"
#endif /* __ARM_FEATURE_SVE */

#include <stdlib.h>
#include "definitions.h"
#include "matrix.h"

static inline int8x16_t boxcarEpi8(int8x16_t u) {

    static const int8x16_t Z = {
            1, -1, 1, -1, 1, -1, 1, -1,
            1, -1, 1, -1, 1, -1, 1, -1
    };
    static const uint8x16_t mask = {
            2, 3, 0, 1, 6, 7, 4, 5,
            10, 11, 8, 9, 14, 15, 12, 13
    };

    u = vmulq_s8(u, Z);
    int8x16_t v = vqtbl1q_s8(u, mask);
    return vaddq_s8(u, v);
}

static int8x16_t convert_epu8_epi8(uint8x16_t u) {

    static const int8x16_t Z = {
            -127, -127, -127, -127, -127, -127, -127, -127,
            -127, -127, -127, -127, -127, -127, -127, -127
    };
    return vaddq_s8(vreinterpretq_s8_u8(u), Z);
}

static inline void convert_epi8_fp16(int8x16_t in, float16x8_t *ret) {

    int16x8_t v = vmovl_high_s8(in),  // int8->int16 (high)
    u = vmovl_s8(vget_low_s8(in));       // int8->int16 (low)
//    int32x4_t   w1 = vmovl_high_s16(u),
//                w0 = vmovl_s16(vget_low_s16(u));
//    ret[0] = vcvtq_f32_s32(w0);
//    ret[1] = vcvtq_f32_s32(w1);


    ret[0] = vcvtq_f16_s16(u);
    ret[1] = vcvtq_f16_s16(v);
//    w1 = vmovl_high_s16(v);
//    w0 = vmovl_s16(vget_low_s16(v));
//
//    ret[2] = vcvtq_f32_s32(w0);
//    ret[3] = vcvtq_f32_s32(w1);
}

static inline void preNormMult(float16x8_t *u, float16x8_t *v) {

    // *v = _mm256_permute_ps(*u, 0xEB);
    float32x2x4_t temp = *(float32x2x4_t *) u;
    temp.val[0] = vdup_lane_f32(temp.val[0], 1);
    temp.val[1] = vdup_lane_f32(temp.val[1], 1);
    float32x4_t w = vreinterpretq_f32_f16(vrev64q_f16(*(float16x8_t *) &temp));
    temp.val[2] = vzip1_f32(vget_low_f32(w), temp.val[0]);
    temp.val[3] = vzip1_f32(vget_high_f32(w),temp.val[1]);
    *v = *(float16x8_t*)(&temp.val[2]);
    // *u = _mm256_permute_ps(*u, 0x5)
    // 0x5 = 1100_4
    // for (a,b,c,d,e,f,g,h) -> (b,b,a,a,f,f,e,e)
//    temp = *(float32x2x4_t *) u;
//    temp.val[0] = vdup_lane_f32(temp.val[0], 0);
//    temp.val[1] = vdup_lane_f32(temp.val[1], 0);
    temp.val[2] = vreinterpret_f32_f16(vdup_laneq_f16(*u, 5));
    temp.val[3] = vreinterpret_f32_f16(vdup_laneq_f16(*u, 4));
    temp.val[0] = vreinterpret_f32_f16(vdup_laneq_f16(*u, 1));
    temp.val[1] = vreinterpret_f32_f16(vdup_laneq_f16(*u, 0));
    temp.val[0] = vzip1_f32(temp.val[0], temp.val[1]);
    temp.val[1] = vzip1_f32(temp.val[2], temp.val[3]);
    *u = *(float16x8_t *)&temp;


    // *u = _mm256_mul_ps(*u, *v);
}

//static inline void preNormAddSubAdd(float16x8_t *u, float16x8_t *v, float16x8_t *w) {
//
//    *w = _mm256_permute_ps(*u, 0x8D);
//    *u = _mm256_addsub_ps(*u, *w);
//    *v = _mm256_mul_ps(*u, *u);
//    *w = _mm256_permute_ps(*v, 0x1B);
//    *v = _mm256_add_ps(*v, *w);
//}

static float fmDemod(float16x8_t *M) {
//    static const float16x8_t all64s = {64.f, 64.f, 64.f, 64.f, 64.f, 64.f, 64.f, 64.f};
//    static const float16x8_t all23s = {23.f, 23.f, 23.f, 23.f, 23.f, 23.f, 23.f, 23.f};
//    static const float16x8_t all41s = {41.f, 41.f, 41.f, 41.f, 41.f, 41.f, 41.f, 41.f};

    float16x8_t w,// y,
    u = M[0];
//            v = M[1];
    preNormMult(&u, &w);
//    preNormAddSubAdd(&u, &v, &w);
    w = w;
    return 0.f;
}

static inline void demodEpi8(uint8x16_t buf, float *__restrict__ result) {

    static const int8x16_t negateBIm = {
            1, 1, 1, -1, 1, 1, 1, -1,
            1, 1, 1, -1, 1, 1, 1, -1
    };
    static const uint8x16_t indexLo = {
            0, 1, 2, 3, 4, 5, 2, 3,
            4, 5, 6, 7, 8, 9, 6, 7
    };
    static const uint8x16_t indexHi = {
            8, 9, 10, 11, 12, 13, 11, 12,
            12, 13, 14, 15, 16, 17, 14, 15
    };
    static int8x16_t prev = {
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0
    };

    float16x8_t M[4];
    int8x16_t lo, hi, u = convert_epu8_epi8(buf);
    u = boxcarEpi8(u);
    hi = vqtbl1q_s8(u, indexHi);
    hi = vmulq_s8(hi, negateBIm);
    lo = vqtbl1q_s8(u, indexLo);
    lo = vmulq_s8(lo, negateBIm);

    prev[12] = lo[0];
    prev[13] = lo[1];

    convert_epi8_fp16(prev, M);

    fmDemod(M);
    // TODO demodulate high half

    convert_epi8_fp16(lo, M);

    // TODO demodulate low half
    // TODO demodulate high half

    prev = hi;
}

void *processMatrix(void *ctx) {

    size_t i, j;
    consumerArgs *args = ctx;
    void *buf __attribute__((aligned(16))) = calloc(DEFAULT_BUF_SIZE, 1);
    float result[DEFAULT_BUF_SIZE >> 4];

    while (!args->exitFlag) {
        sem_wait(&args->full);
        pthread_mutex_lock(&args->mutex);
        memcpy(buf, args->buf, DEFAULT_BUF_SIZE);
        pthread_mutex_unlock(&args->mutex);
        sem_post(&args->empty);

        for (i = 0, j = 0; i < DEFAULT_BUF_SIZE; i += 16, j += 4) {
            demodEpi8(*(uint8x16_t *) (buf + i), result + j);

            if (args->gain) {
                vmulq_n_f32(*(float32x4_t *) result, args->gain);
            }
        }
        fwrite(result, sizeof(float), DEFAULT_BUF_SIZE >> 3, args->outFile);
    }
    return NULL;
}

void allocateBuffer(void **buf, const size_t len) {

    *buf = calloc(len, 1);
}