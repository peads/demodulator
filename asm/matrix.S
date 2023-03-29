#ifdef __clang__
    .globl  _readFile
    _readFile:
#else
    .globl  readFile
    readFile:
#endif
    .text
    pushq %rbp
    movq %rsp, %rbp
//    addq $-8, %rsp
    pushq %rbx
    pushq %r12
    pushq %r13
    pushq %r14
    pushq %r15
    pushq (%rdx)
    pushq %r9
    pushq (%r8)

    shlq $4, %rsi
    addq %rsi, %rdx   // store address of end of array
    negq %rsi

    leaq (%rcx), %r15 // file
    movq %rdx, %r14   // buf128
    movq %rsi, %r12   // len
    movq %rdi, %rbx   // buf
L7: 
    xorq %r13, %r13       // ret = 0
L6: 
    movq %rbx, %rdi    // set the fread arguments, args->file
    movq $1, %rsi
    movq $4, %rdx
    movq %r15, %rcx
    call _fread               // TODO change macro-ifdef-defined label
    addq %rax, %r13

    movq %r15, %rdi
    callq _checkFileStatus    // TODO consider inlining file err/eof check
                                    // TODO change macro-ifdef-defined label

    vmovaps (%rbx), %xmm1 // args->buf
    vpaddb all_nonetwentysevens(%rip), %xmm1, %xmm1
    vpmovsxbw %xmm1, %xmm1
    vpmovsxwd %xmm1, %xmm1
    vcvtdq2ps %xmm1, %xmm1
    vmulps cnj_transform(%rip), %xmm1, %xmm1
    vmovq %xmm0, %rcx   // args->squelch
    test %rcx, %rcx        // if squelch != NULL
    jz nosquelch          // apply squelch
    vmulps %xmm1, %xmm1, %xmm0
    vpermilps $0xB1, %xmm0, %xmm3
    vaddps %xmm0, %xmm3, %xmm0
    vmulps all_hundredths(%rip), %xmm0, %xmm0
    vcmpps $0x1D, %xmm1, %xmm0, %xmm0
    vandps %xmm0, %xmm1, %xmm1
nosquelch:
    vmovaps %xmm1, (%r12, %r14)
    addq $16, %r12
    jl L6

    test %r13, %r13
    jz loopEnd

    movb (%rsp), %cl
    testb %cl, %cl
    jz isNotRdc           // if isRdc
// removeDCSpike(buf128, DEFAULT_BUF_SIZE);
    movq %r14, %rdi
    movl $1024, %esi
    callq _removeDCSpike // TODO change to actual label name once in own .S file
isNotRdc: 
//    movb 1(%rsp), %cl
//    testb %cl, %cl
//    jnz isOt           // if !isOt
// applyComplexConjugate(buf128, DEFAULT_BUF_SIZE);
//    movq %r14, %rdi
//    movl $1024, %esi
//    callq _applyComplexConjugate // TODO change to actual label name once in own .S filejm
//isOt: 
// filter(buf128, DEFAULT_BUF_SIZE, downsample);
    movq %r14, %rdi
    movl $1024, %esi
    movb 2(%rsp), %dl
    test %dl, %dl
    jz end
    callq _filter // TODO change to actual label
    jmp end
// demodulateFmData(buf128, depth, result);
    movq %r14, %rdi
    movl %eax, %esi
    leaq result(%rip), %rdx
    callq _demodulateFmData // TODO change to actual label
// fwrite(result, OUTPUT_ELEMENT_BYTES, depth, outFile);
    movq result(%rip), %rdi // TODO result put in header.S
    movl $1024, %esi
    movq %rax, %rdx
    movq (%rsp), %rdi
    callq _fwrite // TODO change macro-ifdef-defined label
loopEnd: 
    movq _exitFlag(%rip), %rcx
    test %rcx, %rcx
    jz L7

end: 
    addq $24, %rsp
//    popq %rdx
//    popq %r9
//    popq %r8
    popq %r15
    popq %r14
    popq %r13
    popq %r12
    popq %rbx
    popq %rbp
    ret