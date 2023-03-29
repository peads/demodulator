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

/*
 * Takes packed floats representing two sets of complex numbers
 * of the form (ar + iaj), (br + ibj), s.t. z = {ar, aj, br, bj}
 * and returns the arguments of the phasors (i.e. Arg[(ar + iaj).(br + ibj)])
 * as a--effectively--a packed float. Doesn't account for special
 * case x<0 && y==0, but this doesn't seem to negatively affect performance.
 */
#ifdef __clang__
    .globl _demodulateFmData, _arg
#define FUN _demodulateFmData:
#else
    .globl demodulateFmData, _arg
#define FUN demodulateFmData:
#endif
_arg:
.text
    vblendps $0b0011, %xmm1, %xmm0, %xmm1
    vinsertf128 $1, %xmm1, %ymm0, %ymm0
    vmulps negate_b_im(%rip), %ymm0, %ymm0 // (ar, aj, br, -bj)

    vpermilps $0xEB, %ymm0, %ymm1     // (ar, aj, br, bj) => (aj, aj, ar, ar)
    vpermilps $0x5, %ymm0, %ymm0      // and                 (bj, br, br, bj)

    vmulps %ymm1, %ymm0, %ymm0        // aj*bj, aj*br, ar*br, ar*bj
    vpermilps $0x8D, %ymm0, %ymm2     // aj*br, aj*bj, ar*bj, ar*br
    vaddsubps %ymm2, %ymm0, %ymm0     //  ... [don't care], ar*bj + aj*br, ar*br - aj*bj, [don't care] ...
    vmulps %ymm0, %ymm0, %ymm1        // ... , (ar*bj + aj*br)^2, (ar*br - aj*bj)^2, ...
    vpermilps $0x1B, %ymm1, %ymm2
    vaddps %ymm2, %ymm1, %ymm1        // ..., (ar*br - aj*bj)^2 + (ar*bj + aj*br)^2, ...

    vrsqrtps %ymm1, %ymm1             // ..., 1/Sqrt[(ar*br - aj*bj)^2 + (ar*bj + aj*br)^2], ...
    vmulps %ymm1, %ymm0, %ymm0        // ... , zj/||z|| , zr/||z|| = (ar*br - aj*bj) / Sqrt[(ar*br - aj*bj)^2 + (ar*bj + aj*br)^2], ...

    vmulps all_64s(%rip), %ymm0, %ymm2        // 64*zj
    vmulps all_23s(%rip), %ymm0, %ymm3        // 23*zr
    vaddps all_41s(%rip), %ymm3, %ymm3        // 23*zr + 41
    vpermilps $0x1B, %ymm3, %ymm3
    vrcpps %ymm3, %ymm3
    vmulps %ymm3, %ymm2, %ymm0

    vextractf128 $1, %ymm0, %xmm1
    vpermilps $1, %ymm0, %ymm0
    vblendps $1, %xmm0, %xmm1, %xmm0
    vcmpps $0x0, %xmm0, %xmm0, %xmm1  // effectively the NAN check
    vandps %xmm1, %xmm0, %xmm0
    vmovq %xmm0, %rax
    jmpq *%rdx

FUN
    movq %rdi, %rcx   // store buf address
    movq %rsi, %r8    // store n
    shlq $4, %r8
    addq %r8, %rcx    // store address of end of buf

    movq %rdx, %r9    // store result address
    shrq %r8
    movq %r8, %r10
    shlq %r8
    addq %r10, %r9     // store address of end of result

    negq %r8
    negq %r10
L4: 
    vmovaps (%rcx,%r8), %xmm1
    vmovaps -16(%rcx,%r8), %xmm0
    leaq (%rip), %rdx
    addq $9, %rdx
    jmp _arg
    movq %rax, (%r9,%r10)
// loop unroll one
    vmovaps 16(%rcx,%r8), %xmm1
    vmovaps (%rcx,%r8), %xmm0
    leaq (%rip), %rdx
    addq $9, %rdx
    jmp _arg
    movq %rax, 8(%r9,%r10)
// loop unroll two
    vmovaps 32(%rcx,%r8), %xmm1
    vmovaps 16(%rcx,%r8), %xmm0
    leaq (%rip), %rdx
    addq $9, %rdx
    jmp _arg
    movq %rax, 16(%r9,%r10)
// loop unroll three
    vmovaps 48(%rcx,%r8), %xmm1
    vmovaps 32(%rcx,%r8), %xmm0
    leaq (%rip), %rdx
    addq $9, %rdx
    jmp _arg
    movq %rax, 24(%r9,%r10)

    vmovaps 64(%rcx,%r8), %xmm1
    vmovaps 48(%rcx,%r8), %xmm0
    leaq (%rip), %rdx
    addq $9, %rdx
    jmp _arg
    movq %rax, 32(%r9,%r10)

    vmovaps 80(%rcx,%r8), %xmm1
    vmovaps 64(%rcx,%r8), %xmm0
    leaq (%rip), %rdx
    addq $9, %rdx
    jmp _arg
    movq %rax, 40(%r9,%r10)

    vmovaps 96(%rcx,%r8), %xmm1
    vmovaps 80(%rcx,%r8), %xmm0
    leaq (%rip), %rdx
    addq $9, %rdx
    jmp _arg
    movq %rax, 48(%r9,%r10)

    vmovaps 112(%rcx,%r8), %xmm1
    vmovaps 96(%rcx,%r8), %xmm0
    leaq (%rip), %rdx
    addq $9, %rdx
    jmp _arg
    movq %rax, 56(%r9,%r10)

    addq $64, %r10 // i++
    addq $128, %r8 // j += 2
    jl L4
    shlq $1, %rsi
    movq %rsi, %rax
    ret