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
#ifdef __clang__
    .globl  _processMatrix
    _processMatrix:
#else
    .globl  processMatrix
    processMatrix:
#endif
    .text
    pushq %rbp
    movq %rsp, %rbp
    //addq $-8, %rsp
    pushq %rbx
    pushq %r12
    pushq %r13
    pushq %r14
    pushq %r15
    pushq %r9

    shlq $4, %rsi
    //pushq buf128(%rip)    // buf128
    pushq %rsi
    pushq (%r8)

    movq %rdi, %rbx   // buf
    leaq (%rcx), %r15 // *inFile
    movq %rsi, %rbp   // len
    vxorps %xmm4, %xmm4, %xmm4  // dc_avg_iq
L7:
    leaq buf128(%rip), %r14   // buf128
    xorq %r12, %r12       // i
    xorq %r13, %r13       // ret = 0
L6: 
    movq %rbx, %rdi    // set the fread arguments, args->file
    movq $1, %rsi
    movq $4, %rdx
    movq %r15, %rcx
    call _fread               // TODO change macro-ifdef-defined label
    addq %rax, %r13

    movq %r15, %rdi
    callq _checkFileStatus          // TODO consider inlining file err/eof check
                                    // TODO change macro-ifdef-defined label
    movb %al, exit_flag(%rip)

    vmovaps (%rbx), %xmm1 // args->buf
    vpaddb all_nonetwentysevens(%rip), %xmm1, %xmm1 // convert uint8 to int16
    vpmovsxbw %xmm1, %xmm1
    vpmovsxwd %xmm1, %xmm1
    vcvtdq2ps %xmm1, %xmm1                  // int16 to float
    vmulps cnj_transform(%rip), %xmm1, %xmm1// ar, -aj, br, -bj // TODO consider storing in xmm5+
    vmovq %xmm0, %rcx                       // args->squelch
    test %rcx, %rcx
    jz nosquelch                            // if squelch > 0
    vmulps %xmm1, %xmm1, %xmm2              // apply squelch
    vpermilps $0xB1, %xmm2, %xmm3
    vaddps %xmm2, %xmm3, %xmm2
    vmulps all_hundredths(%rip), %xmm2, %xmm2 // TODO consider storing in xmm5+
    vcmpps $0x1D, %xmm1, %xmm2, %xmm2
    vandps %xmm2, %xmm1, %xmm1
nosquelch:
    movb (%rsp), %cl
    testb %cl, %cl
    jz nordc                                // if rdc
    vsubps %xmm4, %xmm1, %xmm2              // apply raw dc iir filter
    vmulps dc_raw_const(%rip), %xmm2, %xmm2 // TODO consider storing in xmm5+
    vaddps %xmm2, %xmm2, %xmm2
    vsubps %xmm2, %xmm1, %xmm1
    vmovaps %xmm2, %xmm4    // can we optimize this out with using xmm4 in place of xmm2 above?
nordc:
    // window filter
    movq %r12, %rcx
    shrq $5, %rcx
    shlq $4, %rcx
    vpermilps $0x4E, %xmm1, %xmm2
    vaddps %xmm2, %xmm1, %xmm1
    vmovaps %xmm1, (%r14, %rcx)

    xorq %rcx, %rcx // reset bits of rcx, s.t. the upper doesn't interfere with squelch check
    addq $16, %r12
    test %r12, %rbp
    jz L6
    test %r13, %r13
    jz loopEnd
// demodulateFmData(buf128, depth, result);
    movq %r14, %rdi
    movl $1024, %esi
    leaq result(%rip), %rdx
    callq _demodulateFmData
// fwrite(result, OUTPUT_ELEMENT_BYTES, depth, outFile);
    leaq result(%rip), %rdi
    movl $4, %esi
    movl $1024, %edx
    movq 16(%rsp), %rcx
    callq _fwrite // TODO change macro-ifdef-defined label
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