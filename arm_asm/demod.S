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
    str     x30, [sp, #-16]!        // store the link reg
    stp     x20, x19, [sp, #-32]!   // store callee-preserved regs

    mov     x19, x0                 // inFile
// TODO cmp and jmp if filename is NULL remembering to set it to #1 for stdout

    mov     x8, #SYS_OPEN
    mov     w0, #AT_FDCWD
    mov     x1, x2              // file name
    mov     w2, #OPEN_FLAGS     // O_TRUNC | O_CREAT | O_WRONLY
    mov     w3, #OPEN_MODE      // rw-rw-r--
    svc     #0
    cmp     x0, #0;
    b.le    fio_error           // if unable to open file end
    mov     x20, x0             // outFile

    ldr     x0, =buf
    ldr     x1, input_element_bytes
    mov     x2, #MATRIX_WIDTH
    mov     x3, x19
    bl      FREAD

// TODO write calls to FERROR and FEOF

    mov     w8, #SYS_WRITE
    ldr     x1, =buf
    mov     x0, x20        // fd
    mov     w2, #MATRIX_WIDTH         // n * size bytes
    svc     #0
    cmp     x0, #0;
    b.le    fio_error

    mov     w8, #SYS_CLOSE
    mov     x0, x20          // fd
    svc     #0
end:

    ldp     x20, x19, [sp], #32   // reload preserved regs
    ldr     x30, [sp], #16  // reload lr
    ret
fio_error:
    mov     w8, #SYS_WRITE
    mov     x0, #1
    ldr     x1, =MSG
    ldr     x2, =LEN
    b       end
coolNumber:
    //leaf functions do not need to push the lr

    //and     w2, w2, #0xFFFF
    //uxth    w2, w2
    //ucvtf   s2, w2

    //and     w3, w3, #0xFF
    sxtb    w1, w1
    ucvtf   s1, w1

    fadd    s0, s1, s0

    ret

.data
RIGHT:
//    .hword 0x0045
    .word 0xbf147ae1//0x3ed70a3d

LEFT:
    .byte 0x46

MSG:
    .asciz "FUCK.\n"
LEN = . - MSG

input_element_bytes:
    .byte 1