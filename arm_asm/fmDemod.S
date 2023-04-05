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
#if defined(__clang__) || defined(__APPLE__)
    .globl  _fmDemod
_fmDemod:
#else
    .globl  fmDemod
fmDemod:
#endif
    ldr x5, =all_nonetwentysevens
    eor x7, x7, x7
    //eor x11, x11, x11
    //ldr x9, =foo
    //ldr x10, =bar
    ld1 {v5.16b}, [x5]
    add x6, x0, x1
L0:
    ld1 {v3.16b, v4.16b}, [x0]//, #32

    add v3.16b, v5.16b, v3.16b
    add v4.16b, v5.16b, v4.16b

    st1 {v3.16b, v4.16b}, [x2]//, #32
    add x0, x0, #16
    add x7, x7, #1
    lsr x8, x7, #1
    lsl x8, x8, #3
    add x2, x2, x8
//    str x8, [x9]
//    add x11, x11, #16
//    str x11, [x10]
    
    cmp x0, x6
    bne L0

    //ldp q3, q4, [x0]

    //saddl v3.8h, v5.8b, v3.8b
    //saddl v4.8h, v6.8b, v4.8b
    //saddl v3.4s, v3.4h, v3.4h

    //scvtf v3.4h, v3.4h
    //scvtf v4.8h, v4.8h

    //stp q3, q4, [x2]
    ret