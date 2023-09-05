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
#include "asm.h"
    .text
    str     x30, [sp, #-0x10]!        // store the link reg
    stp     x20, x19, [sp, #-0x20]!   // store callee-preserved regs
    stp     x22, x21, [sp, #-0x20]!   // store callee-preserved regs
    stp     x24, x23, [sp, #-0x20]!   // store callee-preserved regs
    stp     x26, x25, [sp, #-0x20]!   // store callee-preserved regs
    sub     sp, sp, #STACK_SIZE

    mov     x19, x0                 // inFile
    mov     x20, #1                 // outFile is stdout if NULL is passed in
    mov     x21, #2                 // input_element_bytes
    sub     x21, x21, x1            // subtract mode from input_element_bytes 
    ldr     x4, =cnj_transform
    ld1     {v22.4s}, [x4]
    add     x23, sp, #STACK_SIZE    // end of buffer
    eor     v25.16b, v25.16b, v25.16b
    ldr     x4, =negate_b_im
    ld1     {v26.4s}, [x4]

    cbz     x2, L7
    mov     x8, #SYS_OPEN
    mov     w0, #AT_FDCWD
    mov     x1, x2              // file name
    mov     w2, #OPEN_FLAGS     // O_TRUNC | O_CREAT | O_WRONLY
    mov     w3, #OPEN_MODE      // rw-rw-r--
    svc     #0
    mov     x20, x0             // outFile
L7:
    mov x24, sp
L6:
    mov     x0, x24
    mov     x1, x21           // input_element_bytes
    mov     x2, #MATRIX_WIDTH // * WIDTH = total bytes read each iteration
    mov     x3, x19
    bl      FREAD

    mov     x0, x19
    bl      FERROR
    cbz     x0, check
    eor     x2, x2, x2
    bl      PERROR
    mov     x0, #1
    b       end
check:
    mov     x0, x19
    bl      FEOF
    cbz     x0, continue
    b       end
continue:
    ld1     {v0.4h}, [x24]
    sshll   v0.4s, v0.4h, #0
    scvtf   v0.4s, v0.4s
    // conj
    fmul    v0.4s, v22.4s, v0.4s

    // swap
    umov    x0, v0.d[1]
    umov    x1, v0.d[0]
    fmov    d1, x0
    ins     v1.d[1], x1
    // boxcar
    fadd    v1.4s, v1.4s, v0.4s
    
    // it's a surprise tool that will help us later
    mov     v0.16b, v25.16b
    mov     v25.16b, v1.16b

    // arg
    // v1 = {ar, aj, br, bj} = {1,2,3,-4}
    // v2 = {dr, dj, br, bj} = {5,6,3,-4}
    umov    x0, v0.d[1]
    ins     v1.d[1], x0
    fmul    v1.4s, v26.4s, v1.4s
    fmul    v0.4s, v26.4s, v0.4s

    // TODO

    // v1_1, v1_2 = {aj, aj, ar, ar}, {dj, dj, dr, dr}
    // v2_1, v2_2 = {bj, br, br, bj}, {bj, br, br, bj} 
    trn1    v2.4s, v0.4s, v0.4s // => {ar, ar, br, br}
    trn2    v3.4s, v0.4s, v0.4s // => {aj, aj, bj, bj}
    umov    x0, v2.d[0]
    ins     v3.d[1], x0

    

    // v1_1, v1_2 
    // = {aj*bj, aj*br, ar*br, ar*bj}, {dj*bj, dj*br, dr*br, dr*bj}
    // = {abjj, abjr, abrr, abrj}, {dbjj, dbjr, dbrr, dbrj}
    // = {-8, 6, 3, -4}, {-24, 18, 15, -20}

    // u = {abjr, abrj, abjj, abrr}, {dbjr, dbrj, dbjj, dbrr}
    // = {6, -4, -8, 3, 18, -20, -24, 15}
    
    // u = {-, +, -, +}
    // = {abjj + abjr, abjr - abrj, abrr + abjj, abrj - abrr},
    //      {dbjj + dbjr, dbjr - dbrj, dbrr + dbjj, dbrj - dbrr} 
    // = {-14, 2, 11, -1}, {-42, -2, 39, -5}

    // v2 = u*u = {196, 4, 121, 1}, {1764, 4, 1521, 25}
    // reverse u =  {1, 121, 4, 196}, {25, 1521, 4, 1764}

    // v2 = v2 + u
    // {197, 125, 125, 197}, {1789, 1525, 1525, 1789}

    // v2 = 1/sqrt(v2)

    // v1 = u/sqrt(v2)

    // u = 64*u/sqrt(v2)

    // v1 = 23*v1 + 41

    // v4 =  reverse v1

    // v4 = 1/v4

    // v1 = u/v1

    // v2 = extract top half of v1

    // v1 = move second element in each lane of v1 to first position

    // v1 = insert first element of v1 into first position of v2

    // v2 = compare v1 to itself for eqaulity (NaN check)

    // v1 = and v1 with v2

    // return first element of v1

    // store the result
    st1     {v0.4s}, [x24]
    add     x24, x24, #16

    cmp     x23, x24
    b.ne    L6

    mov     w8, #SYS_WRITE
    mov     x0, x20             // fd
    mov     x1, sp
    mov     x2, #STACK_SIZE       // output_element_bytes
    svc     #0
    cbnz    x0, L7

    cmp     x20, #1
    b.eq    end
    mov     w8, #SYS_CLOSE
    mov     x0, x20                 // fd
    svc     #0
end:
    add     sp, sp, #STACK_SIZE
    ldp     x26, x25, [sp], #0x20   // reload preserved regs
    ldp     x24, x23, [sp], #0x20   // reload preserved regs
    ldp     x22, x21, [sp], #0x20   // reload preserved regs
    ldp     x20, x19, [sp], #0x20   // reload preserved regs
    ldr     x30, [sp], #0x10        // reload lr
    ret
