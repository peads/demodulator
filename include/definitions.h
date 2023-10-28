/*
 * This file is part of the demodulator distribution
 * (https://github.com/peads/demodulator).
 * and code originally part of the misc_snippets distribution
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

#ifndef DEMODULATOR_DEFINITIONS_H
#define DEMODULATOR_DEFINITIONS_H

#ifndef DEFAULT_BUF_SIZE
#define DEFAULT_BUF_SIZE 262144L
#endif

#define WC 0.00133333333f
#define CDAB_INDEX _MM_SHUFFLE(1,0,3,2)
//2,3,0,1)

#if (!(defined(NO_INTRINSICS) || defined(NO_AVX2)) && (defined(__AVX__) || defined(__AVX2__)))
#define ALIGNMENT 32
#define HAS_AVX
#endif

#if (!(defined(NO_INTRINSICS) || defined(NO_AVX512)) && defined(__AVX512BW__) && defined(__AVX512F__) && defined(__AVX512DQ__))
#ifdef ALIGNMENT
#undef ALIGNMENT
#define ALIGNMENT 64
#endif
#ifndef HAS_AVX
#define HAS_AVX
#endif
#define HAS_AVX512
#endif

#if (defined(__SSE__) || defined(__SSE2__) || defined(__SSE3__) || defined(__SSE4_1__) \
        || defined(__SSE4_2__) || defined(__SSE_MATH__) || defined(__SSE2_MATH__) \
        || defined(__SSSE3__))
#define HAS_SSE
#endif

#if defined(HAS_AVX) || defined(HAS_SSE)
#define HAS_EITHER_AVX_OR_SSE
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#define HAS_AARCH64
#endif

/**
 * Takes a 4x4 matrix and applies it to a 4x1 vector.
 * Here, it is used to apply the same rotation matrix to
 * two complex numbers. i.e., for the the matrix
 * T = {{a,b}, {c,d}} and two vectors {u1,u2} and {v1,v2}
 * concatenated, s.t. u = {u1,u2,v1,v2}, Tu =
 * {a*u1 + c*u1, b*u2 + d*u2, ... , b*v2 + d*v2}
 *
__asm__(
#ifdef __clang__
"_apply4x4_4x1Transform: "
#else
"apply4x4_4x1Transform: "
#endif
    "vmulps 16(%rdi), %xmm0, %xmm2\n\t"      // u1*a11, u2*a12, u3*a13, ...
    "vmulps (%rdi), %xmm0, %xmm1\n\t"        // u1*a21, u2*a22, ...
    "vpermilps $0xB1, %xmm2, %xmm0\n\t"
    "vaddps %xmm2, %xmm0, %xmm2\n\t"         // u1*a11 + u2*a12, ... , u3*a13 + u4*a14
    "vpermilps $0xB1, %xmm1, %xmm0\n\t"
    "vaddps %xmm1, %xmm0, %xmm1\n\t"         // u1*a21 + u2*a22, ... , u3*a23 + u4*a24
    "vblendps $0xA, %xmm2, %xmm1, %xmm0\n\t" // u1*a11 + u2*a12, u1*a21 + u2*a22,
    "ret"                                    // u3*a13 + u4*a14, u3*a23 + u4*a24
);
*/
//struct rotationMatrix {
//    const __m128 a1;
//    const __m128 a2;
//};
//static const struct rotationMatrix CONJ_TRANSFORM = {
//        {1, 0, 1, 0},
//        {0, -1, 0, -1}
//};
//static inline struct rotationMatrix generateRotationMatrix(const float theta, const float phi) {
//
//    const float cosT = cosf(theta);
//    const float sinP = sinf(phi);
//
//    struct rotationMatrix result = {
//        .a1 = {cosT, -sinP, cosT, -sinP},
//        .a2 = {sinP, cosT, sinP, cosT}
//    };
//
//    return result;
//}
#endif //DEMODULATOR_DEFINITIONS_H
