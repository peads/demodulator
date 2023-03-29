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
    .globl  _processMatrix
    _processMatrix:
#else
#define FWRITE fwrite
#define FREAD fread
#define STATUS checkFileStatus
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

    movw $DEFAULT_BUF_SIZE, %bp
    shlw $4, %bp

    pushq %rdx
    pushq %rbp
    pushq (%rsi)

    leaq _buf(%rip), %rbx   // buf
    leaq (%rdi), %r15 // *inFile
    vxorps %xmm4, %xmm4, %xmm4  // dc_avg_iq
    vmovaps dc_raw_const(%rip), %xmm5
    vmovaps all_hundredths(%rip), %xmm6
    vmovaps cnj_transform(%rip), %xmm7
L7:
    leaq _buf128(%rip), %r14
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
    callq STATUS
    movb %al, exit_flag(%rip)

    vmovaps (%rbx), %xmm1
    vpaddb all_nonetwentysevens(%rip), %xmm1, %xmm1 // convert uint8 to int16
    vpmovsxbw %xmm1, %xmm1
    vpmovsxwd %xmm1, %xmm1
    vcvtdq2ps %xmm1, %xmm1                  // int16 to float
    vmulps %xmm7, %xmm1, %xmm1              // ar, -aj, br, -bj
    vmovq %xmm0, %rcx                       // args->squelch
    test %rcx, %rcx
    jz nosquelch                            // if squelch > 0
    vmulps %xmm1, %xmm1, %xmm2              // apply squelch
    vpermilps $0xB1, %xmm2, %xmm3
    vaddps %xmm2, %xmm3, %xmm2
    vmulps %xmm6, %xmm2, %xmm2
    vcmpps $0x1D, %xmm1, %xmm2, %xmm2
    vandps %xmm2, %xmm1, %xmm1
nosquelch:
    movb (%rsp), %cl
    testb %cl, %cl
    jz nordc                                // if rdc
    vsubps %xmm4, %xmm1, %xmm4              // apply raw dc iir filter
    vmulps %xmm5, %xmm4, %xmm4
    vaddps %xmm4, %xmm4, %xmm4
    vsubps %xmm4, %xmm1, %xmm1
nordc:
    // window filter
    movq %r12, %rcx
    shrq $5, %rcx
    shlq $4, %rcx
    vpermilps $0x4E, %xmm1, %xmm2
    vaddps %xmm2, %xmm1, %xmm1
    vmovaps %xmm1, (%r14, %rcx)

    xorq %rcx, %rcx // reset bits of rcx, s.t. the upper doesn't interfere with squelch check
    addq $MATRIX_ELEMENT_BYTES, %r12
    test %r12, %rbp
    jz L6
    test %r13, %r13
    jz loopEnd
// demodulateFmData(buf128, depth, result);
    movq %r14, %rdi
    movl $512, %esi //TODO
    leaq _result(%rip), %rdx
    callq _demodulateFmData
// fwrite(result, OUTPUT_ELEMENT_BYTES, depth, outFile);
    leaq _result(%rip), %rdi
    movw $OUTPUT_ELEMENT_BYTES, %si
    movl $DEFAULT_BUF_SIZE, %edx
    movq 16(%rsp), %rcx
    callq FWRITE
loopEnd: 
    movb exit_flag(%rip), %al
    testb %al, %al
    jz L7
end: 
    addq $24, %rsp
    popq %r15
    popq %r14
    popq %r13
    popq %r12
    popq %rbx
    popq %rbp
    ret
