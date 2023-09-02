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

    mov     x19, x0                 // inFile
    mov     x20, #1                 // outFile is stdout if NULL is passed in
    mov     x21, #2                 // input_element_bytes
    sub     x21, x21, x1            // substract mode from  input_element_bytes


    cbz     x2, L7
    mov     x8, #SYS_OPEN
    mov     w0, #AT_FDCWD
    mov     x1, x2              // file name
    mov     w2, #OPEN_FLAGS     // O_TRUNC | O_CREAT | O_WRONLY
    mov     w3, #OPEN_MODE      // rw-rw-r--
    svc     #0
    mov     x20, x0             // outFile
L7:
// TODO zero out loop variables for next iteration
L6:
    ldr     x0, =buf
    mov     x1, x21             // input_element_bytes
    mov     x2, #MATRIX_WIDTH   // * WIDTH = total bytes read each iteration
    mov     x3, x19
    bl      FREAD
    add     x22, x22, x0        // increment index

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
    mov     w8, #SYS_WRITE
    ldr     x1, =buf
    mov     x0, x20             // fd
    mov     w2, #8   // n * output_element_bytes
    svc     #0
    cbnz    x0, L7

    cmp     x20, #1
    b.eq    end
    mov     w8, #SYS_CLOSE
    mov     x0, x20          // fd
    svc     #0
end:
    ldp     x22, x21, [sp], #0x20   // reload preserved regs
    ldp     x20, x19, [sp], #0x20   // reload preserved regs
    ldr     x30, [sp], #0x10  // reload lr
    ret
