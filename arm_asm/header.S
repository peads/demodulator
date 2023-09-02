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

    .globl all_hundredths, all_nonetwentysevens, all_64s, all_23s, all_41s, cnj_transform, dc_raw_const, negate_b_im

    .bss
    .comm buf, MATRIX_WIDTH << 2
    .comm dc_avg_iq, MATRIX_WIDTH << 2
    .comm squelch, MATRIX_WIDTH << 2
    .comm result, DEFAULT_BUF_SIZE << 3
    .comm input_element_bytes, 1

    .data
#ifndef __APPLE__
    .section .rodata
#endif
all_hundredths:
    .rept MATRIX_WIDTH
    .single 0.01
    .endr
all_nonetwentysevens:
    .rept 2
    .quad -0x7f7f7f7f7f7f7f7f
    .endr
all_64s:
    .rept MATRIX_WIDTH << 1
    .single 64.0
    .endr
    .p2align 5
all_23s:
    .rept MATRIX_WIDTH << 1
    .single 23.0
    .endr
    .p2align 5
all_41s:
    .rept MATRIX_WIDTH << 1
    .single 41.0
    .endr
    .p2align 5
cnj_transform:
    .single 1.0, -1.0, 1.0, -1.0//, 1.0, -1.0, 1.0, -1.0
dc_raw_const:
    .rept MATRIX_WIDTH
    .single 1e-05
    .endr
negate_b_im:
    .single 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0

