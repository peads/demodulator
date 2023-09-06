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
    .globl _demodulateFmData, _arg
_arg:
    .text
    vblendps        $0b0011, %xmm1, %xmm0, %xmm1
    vinsertf128     $1, %xmm1, %ymm0, %ymm0
    vmulps          %ymm12, %ymm0, %ymm0 // (ar, aj, br, -bj)

    vpermilps       $0xEB, %ymm0, %ymm1     // (ar, aj, br, bj) => (aj, aj, ar, ar)
    vpermilps       $0x5, %ymm0, %ymm0      // and                 (bj, br, br, bj)

    vmulps          %ymm1, %ymm0, %ymm0        // aj*bj, aj*br, ar*br, ar*bj
    vpermilps       $0x8D, %ymm0, %ymm2     // aj*br, aj*bj, ar*bj, ar*br
    vaddsubps       %ymm2, %ymm0, %ymm0     //  ... [don't care], ar*bj + aj*br, ar*br - aj*bj, [don't care] ...
    vmulps          %ymm0, %ymm0, %ymm1        // ... , (ar*bj + aj*br)^2, (ar*br - aj*bj)^2, ...
    vpermilps       $0x1B, %ymm1, %ymm2
    vaddps          %ymm2, %ymm1, %ymm1        // ..., (ar*br - aj*bj)^2 + (ar*bj + aj*br)^2, ...

    vrsqrtps        %ymm1, %ymm1             // ..., 1/Sqrt[(ar*br - aj*bj)^2 + (ar*bj + aj*br)^2], ...
    vmulps          %ymm1, %ymm0, %ymm0        // ... , zj/||z|| , zr/||z|| = (ar*br - aj*bj) / Sqrt[(ar*br - aj*bj)^2 + (ar*bj + aj*br)^2], ...

    vmulps          %ymm8, %ymm0, %ymm2      // 64*zj
    vfmadd132ps     %ymm9, %ymm10, %ymm0 // 23*zr + 41s
    vpermilps       $0x1B, %ymm0, %ymm3
    vrcpps          %ymm3, %ymm3
    vmulps          %ymm3, %ymm2, %ymm0

    vextractf128    $1, %ymm0, %xmm1
    vpermilps       $1, %ymm0, %ymm0
    vblendps        $1, %xmm0, %xmm1, %xmm0
    vcmpps          $0x0, %xmm0, %xmm0, %xmm1  // effectively the NAN check
    vandps          %xmm1, %xmm0, %xmm0
    vucomiss        %xmm1, %xmm13
    jb              nogain
    vmulps          %xmm13, %xmm0, %xmm0
nogain:
    vmovq           %xmm0, %rax
    jmpq *%rdx
