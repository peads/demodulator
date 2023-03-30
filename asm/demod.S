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
#ifdef __clang__
#define FWRITE _fwrite
#define FREAD _fread
#define STATUS _checkFileStatus
#define FERROR _ferror
#define FEOF _feof
#define PERROR _perror
    .globl  _processMatrix
    _processMatrix:
#else
#define FWRITE fwrite
#define FREAD fread
#define STATUS checkFileStatus
#define FERROR ferror
#define FEOF feof
#define PERROR perror
    .globl  processMatrix
    processMatrix:
#endif
    .text
    pushq %rbp
    pushq %rbx
    pushq %r12
    pushq %r13
    pushq %r14
    pushq %r15

    vbroadcastss %xmm0, %xmm0
    vmovaps %xmm0, squelch(%rip)
    movw $DEFAULT_BUF_SIZE, %bp
    shlw $4, %bp

    pushq %rdx
    pushq %rbp
    pushq (%rsi)

    leaq buf(%rip), %rbx        // buf
    leaq (%rdi), %r15           // *inFile
L7:
    leaq buf128(%rip), %r14
    xorq %r12, %r12       // i
    xorq %r13, %r13
L6:
    movq %rbx, %rdi
    movq $INPUT_ELEMENT_BYTES, %rsi
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
    jmp end
check:
    movq %r15, %rdi
    call FEOF
    testb %al, %al
    jz continue
    movb $-1, %al
    jmp end
continue:
    vmovaps (%rbx), %xmm1
    vpaddb all_nonetwentysevens(%rip), %xmm1, %xmm1 // convert uint8 to int16
    vpmovsxbw %xmm1, %xmm1
    vpmovsxwd %xmm1, %xmm1
    vcvtdq2ps %xmm1, %xmm1                  // int16 to float
    movb 1(%rsp), %cl
    testb %cl, %cl
    jnz oft
    vmulps cnj_transform(%rip), %xmm1, %xmm1              // ar, -aj, br, -bj
oft:
    movaps squelch(%rip), %xmm0
    ptest %xmm0, %xmm0
    jz nosquelch                            // if squelch > 0
    vmulps %xmm1, %xmm1, %xmm2              // apply squelch
    vpermilps $0xB1, %xmm2, %xmm3
    vaddps %xmm2, %xmm3, %xmm2
    vmulps all_hundredths(%rip), %xmm2, %xmm2
    vcmpps $0x1D, %xmm1, %xmm2, %xmm2
    vandps %xmm2, %xmm1, %xmm1
nosquelch:
    movb (%rsp), %cl
    testb %cl, %cl
    jz nordc                                // if rdc
    vsubps dc_avg_iq(%rip), %xmm1, %xmm4    // apply raw dc iir filter
    vmulps dc_raw_const(%rip), %xmm4, %xmm4
    vaddps %xmm4, %xmm4, %xmm4
    vsubps %xmm4, %xmm1, %xmm1
    vmovaps %xmm4, dc_avg_iq(%rip)
nordc:
    // window filter
    movq %r12, %rax
    shrq $5, %rax
    shlq $4, %rax
    vpermilps $0x4E, %xmm1, %xmm2
    vaddps %xmm2, %xmm1, %xmm1
    vmovaps %xmm1, (%r14, %rax)

    addq $MATRIX_ELEMENT_BYTES, %r12
    test %r12, %rbp
    jz L6
// demodulateFmData(buf128, depth, result);
    movq %r14, %rdi
    movl $HALF_BUF_SIZE, %esi
    leaq result(%rip), %rdx
    callq _demodulateFmData
// fwrite(result, OUTPUT_ELEMENT_BYTES, depth, outFile);
    leaq result(%rip), %rdi
    movw $OUTPUT_ELEMENT_BYTES, %si
    movl $DEFAULT_BUF_SIZE, %edx
    movq 16(%rsp), %rcx
    callq FWRITE
    jmp L7
end:
    addq $24, %rsp
    popq %r15
    popq %r14
    popq %r13
    popq %r12
    popq %rbx
    popq %rbp
    ret
