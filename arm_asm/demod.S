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
    stp     x30, x27, [sp, #-0x20]!        // store the link reg
    stp     x20, x19, [sp, #-0x20]!   // store callee-preserved regs
    stp     x22, x21, [sp, #-0x20]!   // store callee-preserved regs
    stp     x24, x23, [sp, #-0x20]!   // store callee-preserved regs
    stp     x26, x25, [sp, #-0x20]!   // store callee-preserved regs
    sub     sp, sp, #(STACK_SIZE << 1)

    mov     x19, x0                 // inFile
    mov     x20, #1                 // outFile is stdout if NULL is passed in
    mov     x21, #2                 // input_element_bytes
    sub     x21, x21, x1            // subtract mode from input_element_bytes 
    add     x23, sp, #STACK_SIZE
    ldr     x4, =all_41s
    ld1     {v19.4s}, [x4]
    ldr     x4, =all_23s
    ld1     {v20.4s}, [x4]
    ldr     x4, =all_64s
    ld1     {v21.4s}, [x4]
    ldr     x4, =cnj_transform
    ld1     {v22.4s}, [x4]
    eor     v25.16b, v25.16b, v25.16b
    ldr     x4, =negate_b_im
    ld1     {v26.4s}, [x4]
    ldr     x4, =addsub
    ld1     {v27.4s}, [x4]

    cbz     x2, L7
    mov     x8, #SYS_OPEN
    mov     w0, #AT_FDCWD
    mov     x1, x2              // file name
    mov     w2, #OPEN_FLAGS     // O_TRUNC | O_CREAT | O_WRONLY
    mov     w3, #OPEN_MODE      // rw-rw-r--
    svc     #0
    mov     x20, x0             // outFile
L7:
    eor     x22, x22, x22
    mov     x24, sp 
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

    // v1_1, v1_2 = {aj, aj, ar, ar}, {dj, dj, dr, dr}
    // v2_1, v2_2 = {bj, br, br, bj}, {bj, br, br, bj} 
    trn1    v3.4s, v0.4s, v0.4s // => {ar, ar, br, br}
    trn2    v2.4s, v0.4s, v0.4s // => {aj, aj, bj, bj}
    umov    x0, v3.d[0]
    ins     v2.d[1], x0         // {aj, aj, ar, ar}

    trn1    v4.4s, v1.4s, v1.4s
    trn2    v3.4s, v1.4s, v1.4s
    umov    x0, v4.d[0]
    ins     v3.d[1], x0         // {dj, dj, dr, dr}

    trn2    v4.2d, v0.2d, v0.2d    
    rev64   v5.4s, v4.4s
    trn2    v4.2d, v5.2d, v4.2d // {bj, br, br, bj}

    // v1_1, v1_2 
    // = {aj*bj, aj*br, ar*br, ar*bj}, {dj*bj, dj*br, dr*br, dr*bj}
    // = {abjj, abjr, abrr, abrj}, {dbjj, dbjr, dbrr, dbrj}
    // = {-8, 6, 3, -4}, {-24, 18, 15, -20}
    fmul    v5.4s, v2.4s, v4.4s
    fmul    v6.4s, v3.4s, v4.4s

    // u = {abjr, abrj, abjj, abrr}, {dbjr, dbrj, dbjj, dbrr}
    // = {6, -4, -8, 3, 18, -20, -24, 15}
    mov     v7.s[0], v5.s[1]
    mov     v7.s[1], v5.s[3]
    mov     v7.s[2], v5.s[0]
    mov     v7.s[3], v5.s[2] 
    mov     v8.s[0], v6.s[1]
    mov     v8.s[1], v6.s[3]
    mov     v8.s[2], v6.s[0]
    mov     v8.s[3], v6.s[2] 
    
    // u = {-, +, -, +}
    // = {abjj + abjr, abjr - abrj, abrr + abjj, abrj - abrr},
    //      {dbjj + dbjr, dbjr - dbrj, dbrr + dbjj, dbrj - dbrr} 
    // = {-14, 2, 11, -1}, {-42, -2, 39, -5}
    fmul    v7.4s, v27.4s,v7.4s
    fadd    v7.4s, v5.4s, v7.4s
    fmul    v8.4s, v27.4s,v8.4s
    fadd    v8.4s, v6.4s, v8.4s
////////////
    // TODO unfuck everything's efficiency
///////////
    // v2 = u*u = {196, 4, 121, 1}, {1764, 4, 1521, 25}
    // reverse u =  {1, 121, 4, 196}, {25, 1521, 4, 1764}
    fmul    v2.4s, v7.4s, v7.4s
    fmul    v3.4s, v8.4s, v8.4s

    mov    v0.d[0], v2.d[1] 
    mov    v0.d[1], v2.d[0]
    
    mov    v1.d[0], v3.d[1] 
    mov    v1.d[1], v3.d[0]

    rev64   v0.4s, v0.4s
    rev64   v1.4s, v1.4s

    // v2 = v2 + u
    // {197, 125, 125, 197}, {1789, 1525, 1525, 1789}
    fadd    v2.4s, v2.4s, v0.4s
    fadd    v3.4s, v3.4s, v1.4s

    // v2 = 1/sqrt(v2)
    frsqrte v4.4s, v2.4s
    frsqrte v5.4s, v3.4s

    // v1 = u/sqrt(v2)
    // {-0.997406006, 0.178863525, 0.98374939, -0.0712432861}, 
    // {-0.992866516, -0.0512084961, 0.998565674, -0.118198395}
    fmul    v4.4s, v4.4s, v7.4s
    fmul    v5.4s, v5.4s, v8.4s

    // u = 64*u/sqrt(v2)
    // {-63.8339844, 11.4472656, 62.9599609, -4.55957031},
    // {-63.543457, -3.27734375, 63.9082031, -7.56469727}
    fmul    v1.4s, v21.4s, v4.4s    // all_64s
    fmul    v2.4s, v21.4s, v5.4s

    // v1 = 23*v1 + 41
    // {18.0596619, 45.1138611, 63.626236, 39.3614044},
    // {18.1640701, 39.8222046, 63.9670105, 38.2814369}
    mov     v8.16b, v19.16b         // all_41s for the acc
    mov     v9.16b, v8.16b

    fmla    v8.4s, v20.4s, v4.4s    // all_23s
    fmla    v9.4s, v20.4s, v5.4s

    // v4 =  reverse v1
    mov     v6.d[0], v8.d[1] 
    mov     v6.d[1], v8.d[0]
    
    mov     v7.d[0], v9.d[1] 
    mov     v7.d[1], v9.d[0]
    
    rev64   v7.4s, v7.4s
    rev64   v6.4s, v6.4s

    // v4 = 1/v4
    // {0.0254020691, 0.0157165527, 0.0221633911, 0.0553741455}, 
    // {0.0261154175, 0.0156364441, 0.0251121521, 0.0550384521}
    // v1 = u/v1
    // {18.0596619, 45.1138611, 63.626236, 39.3614044},
    // {18.1640701, 39.8222046, 63.9670105, 38.2814369}
    frecpe  v6.4s, v6.4s
    frecpe  v7.4s, v7.4s
    fmul    v6.4s, v6.4s, v1.4s
    fmul    v7.4s, v7.4s, v2.4s

    // store the result
    eor     v0.16b, v0.16b, v0.16b 
    mov     v0.s[1], v7.s[1]
    //mov     v0.s[0], v6.s[1]
         
    fcmeq   v1.4s, v0.4s, v0.4s
    and     v0.16b, v0.16b, v1.16b

    lsr     x0, x22, #1
    add     x0, x23, x0
    st1     {v0.4s}, [x0]

    add     x22, x22, #16
    adds    x0, x22, #-(STACK_SIZE >> 1) 
endinner:
    b.ne    L6

    mov     w8, #SYS_WRITE
    mov     x0, x20             // fd
    //mov     x1, sp
    add     x1, sp, #(STACK_SIZE)
    mov     x2, #STACK_SIZE       // output_element_bytes
    svc     #0
    cbnz    x0, L7

    cmp     x20, #1
    b.eq    end
    mov     w8, #SYS_CLOSE
    mov     x0, x20                 // fd
    svc     #0
end:
    add     sp, sp, #(STACK_SIZE << 1)
    ldp     x26, x25, [sp], #0x20   // reload preserved regs
    ldp     x24, x23, [sp], #0x20   // reload preserved regs
    ldp     x22, x21, [sp], #0x20   // reload preserved regs
    ldp     x20, x19, [sp], #0x20   // reload preserved regs
    ldp     x30, x27, [sp], #0x20        // reload lr
    ret
