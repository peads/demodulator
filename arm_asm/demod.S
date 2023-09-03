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
    sub   sp, sp, #(8 * 16)

    mov     x19, x0                 // inFile
    mov     x20, #1                 // outFile is stdout if NULL is passed in
    mov     x21, #2                 // input_element_bytes
    sub     x21, x21, x1            // subtract mode from input_element_bytes
    ldr     x24, =(result + OUT_BUF_SIZE)

    cbz     x2, L7
    mov     x8, #SYS_OPEN
    mov     w0, #AT_FDCWD
    mov     x1, x2              // file name
    mov     w2, #OPEN_FLAGS     // O_TRUNC | O_CREAT | O_WRONLY
    mov     w3, #OPEN_MODE      // rw-rw-r--
    svc     #0
    mov     x20, x0             // outFile
L7:
    ldr     x23, =result
L6:
//    ldr     x0, =buf
    mov     x0, sp
    mov     x1, x21           // input_element_bytes
    mov     x2, #MATRIX_WIDTH   // * WIDTH = total bytes read each iteration
    mov     x3, x19
    bl      FREAD

    mov     x1, #8
    ldr     x0, [sp]
    str     x0, [x23] // input_element_bytes * width ?
    ld1     {v0.h}[0], [x23]
    st1     {v0.h}[0], [x23], x1

//    add     x23, x23, x0        // increment index

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
//    ldr     x0, buf
//    str     x0, [x23]
//    ldr     x0, =buf
//    ld1     {v0.s}[0], [x0]
//    sxtl    v0.4h, v0.8b //  TODO uint8 also use neg 127s for add
//    sxtl    v0.4s, v0.4h    // == sshll v0.4s, v0.4h, #0
//    scvtf   v0.4s, v0.4s
//    scvtf   v0.4h, v0.4h
//    st1     {v0.s}[0], [x23]

//    add x23, x23, #OUTPUT_ELEMENT_BYTES
    cmp     x23, x24
    b.ne    L6

    mov     w8, #SYS_WRITE
    mov     x0, x20             // fd
    ldr     x1, =result
    mov     w2, #OUT_BUF_SIZE   // n * output_element_bytes
    svc     #0
    cbnz    x0, L7

    cmp     x20, #1
    b.eq    end
    mov     w8, #SYS_CLOSE
    mov     x0, x20                 // fd
    svc     #0
end:
    add   sp, sp, #(8 * 16)
    ldp     x26, x25, [sp], #0x20   // reload preserved regs
    ldp     x24, x23, [sp], #0x20   // reload preserved regs
    ldp     x22, x21, [sp], #0x20   // reload preserved regs
    ldp     x20, x19, [sp], #0x20   // reload preserved regs
    ldr     x30, [sp], #0x10        // reload lr
    ret
