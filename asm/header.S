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
#include "definitions.h"
    .globl all_hundredths, all_nonetwentysevens, all_64s, all_23s, all_41s, cnj_transform, dc_avg_iq, dc_raw_const, negate_b_im, exit_flag, result, buf128, squelch//, buf
    .data
all_hundredths:
    .rept 8
    .single 0.01
    .endr
    .p2align 5
all_nonetwentysevens:
    .rept 2
    .quad -0x7f7f7f7f7f7f7f7f
    .endr
    .p2align 5
all_64s:
    .rept 8
    .single 64.0
    .endr
    .p2align 5
all_23s:
    .rept 8
    .single 23.0
    .endr
    .p2align 5
all_41s:
    .rept 8
    .single 41.0
    .endr
    .p2align 5
cnj_transform:
    .rept 2
    .single 1.0, -1.0, 1.0, -1.0//, 1.0, -1.0, 1.0, -1.0
    .endr
    .p2align 5
dc_avg_iq:
    .zero 1
    .p2align 5
squelch:
    .zero 1
    .p2align 5
dc_raw_const:
    .rept 8
    .single 1e-05
    .endr
    .p2align 5
negate_b_im:
    .rept 2
    .single 1.0, 1.0, 1.0, -1.0//, 1.0, 1.0, 1.0, -1.0
    .endr
    .p2align 5
result:
    .zero HALF_BUF_SIZE << 4
    .p2align 4
buf:
    .zero 1 << 5
    .p2align 4
buf128:
    .zero DEFAULT_BUF_SIZE << 5
    .p2align 5
exit_flag:
    .byte 0

