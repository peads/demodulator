    .globl _applyComplexConjugate
_applyComplexConjugate:
    .text
    movq %rdi, %rcx
    movq %rsi, %rax
    shlq $4, %rax
    addq %rax, %rcx
    negq %rax
L5: 
    vmovaps (%rcx,%rax), %xmm0
    vmulps cnj_transform(%rip), %xmm0, %xmm0
    vmovaps %xmm0, (%rcx,%rax)

    vmovaps 16(%rcx,%rax), %xmm0
    vmulps cnj_transform(%rip), %xmm0, %xmm0
    vmovaps %xmm0, 16(%rcx,%rax)

    vmovaps 32(%rcx,%rax), %xmm0
    vmulps cnj_transform(%rip), %xmm0, %xmm0
    vmovaps %xmm0, 32(%rcx,%rax)

    vmovaps 48(%rcx,%rax), %xmm0
    vmulps cnj_transform(%rip), %xmm0, %xmm0
    vmovaps %xmm0, 48(%rcx,%rax)

    vmovaps 64(%rcx,%rax), %xmm0
    vmulps cnj_transform(%rip), %xmm0, %xmm0
    vmovaps %xmm0, 64(%rcx,%rax)

    vmovaps 80(%rcx,%rax), %xmm0
    vmulps cnj_transform(%rip), %xmm0, %xmm0
    vmovaps %xmm0, 80(%rcx,%rax)

    vmovaps 96(%rcx,%rax), %xmm0
    vmulps cnj_transform(%rip), %xmm0, %xmm0
    vmovaps %xmm0, 96(%rcx,%rax)

    vmovaps 112(%rcx,%rax), %xmm0
    vmulps cnj_transform(%rip), %xmm0, %xmm0
    vmovaps %xmm0, 112(%rcx,%rax)

    addq $128, %rax
    jl L5
    ret