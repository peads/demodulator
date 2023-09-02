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
    pushq %rbp
    pushq %rbx
    pushq %r12
    pushq %r13
    pushq %r14
    pushq %r15

    movw $DEFAULT_BUF_SIZE, %bp
    shlw $4, %bp

    pushq %rdx
    pushq %rbp
    pushq %rsi

    leaq buf(%rip), %rbx        // buf
    leaq (%rdi), %r15           // *inFile
    leaq result(%rip), %r14

    movq $2, input_element_bytes(%rip)
    subq %rsi, input_element_bytes(%rip)

    movl $1, 16(%rsp)
    test %rdx, %rdx
    jz L7
    mov $SYS_OPEN, %eax
    mov %rdx, %rdi          // output stream
    mov $OPEN_FLAGS, %esi   // O_TRUNC | O_CREAT | O_WRONLY
    mov $OPEN_MODE, %edx    // rw-rw-r--
    syscall
    mov %rax, 16(%rsp)      // save the fd
L7:
    xorq %r12, %r12         // i
    xorq %r13, %r13
L6:
    movq %rbx, %rdi
    movq input_element_bytes(%rip), %rsi
    movq $MATRIX_WIDTH, %rdx
    movq %r15, %rcx
    call FREAD
    addq %rax, %r13

    movq %r15, %rdi
    call FERROR
    testb %al, %al
    jz check
    xorq %rdx, %rdx
    call PERROR
    mov $1, %rax
    jmp end
check:
    movq %r15, %rdi
    call FEOF
    testb %al, %al
    jz continue
    jmp end
continue:
    vmovdqa (%rbx), %xmm1
    movb (%rsp), %cl
    testb %cl, %cl
    jz int16
    vpaddb all_nonetwentysevens(%rip), %xmm1, %xmm1 // convert uint8 to int16
    vpmovsxbw %xmm1, %xmm1
int16:
    vpmovsxwd %xmm1, %xmm1                          // int16 to float
    vcvtdq2ps %xmm1, %xmm1
    vmulps cnj_transform(%rip), %xmm1, %xmm1        // ar, -aj, br, -bj
    // boxcar filter
    vpermilps $0x4E, %xmm1, %xmm2
    vaddps %xmm2, %xmm1, %xmm1
    movq %r12, %rcx
    shrq $1, %rcx

    vmovaps %xmm5, %xmm0
    vmovaps %xmm1, %xmm5
    leaq (%rip), %rdx
    addq $9, %rdx
    jmp _arg
    movq %rax, (%r14, %rcx)

    addq $MATRIX_ELEMENT_BYTES, %r12
    test %r12, %rbp
    jz L6

    mov $SYS_WRITE, %eax
    mov 16(%rsp), %rdi       // fd
    lea result(%rip), %rsi   // data
    mov $0x4000, %edx        // size = OUTPUT_ELEMENT_BYTES*QTR_BUF_SIZE
    syscall
    test %rax, %rax
    jnz L7
end:
    mov $SYS_CLOSE, %eax
    mov 16(%rsp), %rdi
    syscall

    addq $24, %rsp
    popq %r15
    popq %r14
    popq %r13
    popq %r12
    popq %rbx
    popq %rbp
    ret
