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
#include "../junk/definitions.h"
#ifdef HAS_AARCH64

float frcpf(float x) {
    __asm__ (
        "frecpe %0.2s, %0.2s\n\t"
        : "=w"(x) : "w"(x) :);
    return x;
}

float frsqrtf(float x) {
    __asm__ (
        "frsqrte %0.2s, %0.2s\n\t"
        : "=w"(x) : "w"(x) :);
    return x;
}
#elif defined(HAS_EITHER_AVX_OR_SSE)

float frcpf(float x) {

    __asm__ (
#ifdef HAS_AVX
        "vrcpss %0, %0, %0\n\t"
#else //if HAS_SSE
        "rcpss %0, %0\n\t"
#endif
        : "=x" (x) : "0" (x));
    return x;
}
float frsqrtf(float x) {

    __asm__ (
#ifdef HAS_AVX
        "vrsqrtss %0, %0, %0\n\t"
#else //if HAS_SSE
        "rsqrtss %0, %0\n\t"
#endif
        : "=x" (x) : "0" (x));
    return x;
}
#else
#include <stdint.h>
#include <math.h>
// both of the following are first-order Newton-Raphson approximations
// that have been modified from the original to use a union to prevent
// undefined behavior inherent with lax aliasing
/// taken from https://stackoverflow.com/a/43245460 and modified as above, but
/// also replaces multiplication by the unit-sign value of the input with using
/// the sign-bit to conditionally assign the necessary values, or their negation
float frcpf(float x) {

    union {
        float f;
        uint32_t i;
    } v = {x};
    const uint8_t sgn = signbit(x);

    x = sgn ? -x : x;

    v.i = -v.i + 0x7EF127EA;

    // Efficient Iterative Approximation Improvement in horner polynomial form.
    // Single iteration, Err = -3.36e-3 * 2^(-flr(log2(x)))
    v.f = v.f * (-x * -v.f + 2.f);
    // Second iteration, Err = -1.13e-5 * 2^(-flr(log2(x)))
    // v.f = v.f * ( 4 + w * (-6 + w * (4 - w)));
    // Third Iteration, Err = +-6.8e-8 *  2^(-flr(log2(x)))
    // v.f = v.f * (8 + w * (-28 + w * (56 + w * (-70 + w *(56 + w * (-28 + w * (8 - w)))))));

    return sgn ? -v.f : v.f ;
}
/// John Carmack's Quake fast reciprocal square-root
float frsqrtf(float y) {

    union {
        float f;
        uint32_t i;
    } pun = {y};
    pun.i = -(pun.i >> 1) + 0x5f3759df;
    pun.f *= 0.5f * (-y * pun.f * pun.f + 3.f);

    return pun.f;
}
#endif

