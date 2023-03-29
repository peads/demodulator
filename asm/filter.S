    .globl  _filter
_filter:
    .text
    shlq $4, %rsi
    movq %rdx, %r8
    negq %r8
L1:
    xorq %rax, %rax
L2:
    movq %rax, %rcx
    vpermilps $0x4E, (%rdi, %rax), %xmm0
    vaddps (%rdi, %rax), %xmm0, %xmm0
    shrq $5, %rcx
    shlq $4, %rcx
    vmovaps %xmm0, (%rdi, %rcx)

    vpermilps $0x4E, 16(%rdi, %rax), %xmm0
    vaddps 16(%rdi, %rax), %xmm0, %xmm0
    vmovaps %xmm0, (%rdi, %rax)

    vpermilps $0x4E, 32(%rdi, %rax), %xmm0
    vaddps 32(%rdi, %rax), %xmm0, %xmm0
    shlq %rcx
    vmovaps %xmm0, (%rdi, %rcx)

    addq $32, %rax
    vpermilps $0x4E, 16(%rdi, %rax), %xmm0
    vaddps 16(%rdi, %rax), %xmm0, %xmm0
    vmovaps %xmm0, (%rdi, %rax)

    cmp %rsi, %rax
    jl L2
    addq $1, %r8
    jl L1

    movq %rdx, %rcx
    addq $4, %rcx
    movq %rsi, %rax
    shr %cl, %rax
    ret