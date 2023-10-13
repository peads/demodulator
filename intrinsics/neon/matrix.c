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

static inline void convert_epi8_fp16(int8x16_t in, int16x8_t *ret) {

    int16x8_t v = vmovl_high_s8(in),  // int8->int16 (high)
    u = vmovl_s8(vget_low_s8(in));       // int8->int16 (low)
//    int32x4_t   w1 = vmovl_high_s16(u),
//                w0 = vmovl_s16(vget_low_s16(u));
//    ret[0] = vcvtq_f32_s32(w0);
//    ret[1] = vcvtq_f32_s32(w1);


    ret[0] = u;
    ret[1] = v;
//    ret[0] = vcvtq_f16_s16(u);
//    ret[1] = vcvtq_f16_s16(v);
//    w1 = vmovl_high_s16(v);
//    w0 = vmovl_s16(vget_low_s16(v));
//
//    ret[2] = vcvtq_f32_s32(w0);
//    ret[3] = vcvtq_f32_s32(w1);
}

static inline void preNormMult(int16x8_t *u, int16x8_t *v) {

    // *v = _mm256_permute_ps(*u, 0xEB);
    int32x2x4_t temp = *(int32x2x4_t *) u;
    temp.val[0] = vdup_lane_s32(temp.val[0], 1);
    temp.val[1] = vdup_lane_s32(temp.val[1], 1);
    int32x4_t w = vreinterpretq_s32_s16(
            vrev64q_s16(*(int16x8_t *) &temp));
    temp.val[2] = vzip1_s32(vget_low_s32(w), temp.val[0]);
    temp.val[3] = vzip1_s32(vget_high_s32(w), temp.val[1]);
    *v = *(int16x8_t *) (&temp.val[2]);
    // *u = _mm256_permute_ps(*u, 0x5)
    temp.val[2] = vreinterpret_s32_s16(vdup_laneq_s16(*u, 5));
    temp.val[3] = vreinterpret_s32_s16(vdup_laneq_s16(*u, 4));
    temp.val[0] = vreinterpret_s32_s16(vdup_laneq_s16(*u, 1));
    temp.val[1] = vreinterpret_s32_s16(vdup_laneq_s16(*u, 0));
    temp.val[0] = vzip1_s32(temp.val[0], temp.val[1]);
    temp.val[1] = vzip1_s32(temp.val[2], temp.val[3]);
    *u = *(int16x8_t *) &temp;
    // *u = _mm256_mul_ps(*u, *v);
    *u = vmulq_s16(*u, *v);
}

static inline void preNormAddSubAdd(int16x8_t *u, int16x8_t *v, int16x8_t *w, float16x8x2_t *M) {

    // TODO consider normalizing before we even begin, or just returning f32s
//    *w = _mm256_permute_ps(*u, 0x8D);
    static const uint8x16_t index = {
            2, 3, 6, 7, 0, 1, 4, 5,
            10, 11, 14, 15, 8, 9, 12, 13
    };

    static const uint8x16_t reverse = {
            12,13,14,15,8,9,10,11,
            4,5,6,7,0,1,2,3
    };

    static const int16x8_t altNegate = {
            -1, 1, -1, 1, -1, 1, -1, 1
    };
    *w = vreinterpretq_s16_u8(vqtbl1q_u8(vreinterpretq_u8_s16(*u), index));
//    *u = _mm256_addsub_ps(*u, *w);
    *u = vaddq_s16(*u, vmulq_s16(altNegate, *w));
    M->val[0] = vcvtq_f16_s16(*u);

//    *v = _mm256_mul_ps(*u, *u);
    int32x4x4_t tmp;
    tmp.val[0] = vmovl_s16(vget_low_s16(*u));
    tmp.val[0] = vmulq_s32(tmp.val[0], tmp.val[0]);
    tmp.val[1] = vmovl_high_s16(*u);
    tmp.val[1] = vmulq_s32(tmp.val[1], tmp.val[1]);
//    *w = _mm256_permute_ps(*v, 0x1B);
//    *v = _mm256_add_ps(*v, *w);
//    *w = vrev64q_s16(*v);
    tmp.val[2] = vreinterpretq_s32_u8(vqtbl1q_u8(vreinterpretq_u8_s32(tmp.val[0]), reverse));
    tmp.val[2] = vaddq_s32(tmp.val[0], tmp.val[2]);
    tmp.val[3] = vreinterpretq_s32_u8(vqtbl1q_u8(vreinterpretq_u8_s32(tmp.val[1]), reverse));
    tmp.val[3] = vaddq_s32(tmp.val[1], tmp.val[3]);

    M->val[1] = vcombine_f16(vcvt_f16_f32(vcvtq_f32_s32(tmp.val[2])), vcvt_f16_f32(vcvtq_f32_s32(tmp.val[3])));
}

static float fmDemod(int16x8_t *M) {
    static const float16x8_t all64s = {64.f, 64.f, 64.f, 64.f, 64.f, 64.f, 64.f, 64.f};
    static const float16x8_t all23s = {23.f, 23.f, 23.f, 23.f, 23.f, 23.f, 23.f, 23.f};
    static const float16x8_t all41s = {41.f, 41.f, 41.f, 41.f, 41.f, 41.f, 41.f, 41.f};

    static const uint8x16_t reverse = {
            12,13,14,15,8,9,10,11,
            4,5,6,7,0,1,2,3
    };

    int16x8_t w,// y,
    u = M[0],
    v = M[1];
    float16x8x4_t N;

    preNormMult(&u, &w);
    preNormAddSubAdd(&u, &v, &w, (float16x8x2_t *) &N);
    N.val[1] = vrsqrteq_f16(N.val[1]);
    N.val[0] = vmulq_f16(N.val[0], N.val[1]);

    N.val[2] = vmulq_f16(all64s, N.val[0]);
    N.val[0] = vfmaq_f16(all23s, N.val[0], all41s);
    N.val[3] = vrecpeq_f16(vreinterpretq_f16_u8(vqtbl1q_u8(vreinterpretq_u8_f16(N.val[0]), reverse)));
    N.val[0] = vmulq_f16(N.val[2], N.val[3]);

    return N.val[0][5];
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

    int16x8_t M[4];
    int8x16_t lo, hi, u = heterodyne(buf);
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