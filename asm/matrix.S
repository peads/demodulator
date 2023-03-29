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
    addq $-8, %rsp
    pushq %rbx
    pushq %r12
    pushq %r13
    pushq %r14
    pushq %r15
    pushq %r9

    shlq $4, %rsi
    addq %rsi, %rdx   // store address of end of array
    negq %rsi
    pushq %rdx
    pushq %rsi
    pushq (%r8)

    movq %rdi, %rbx   // buf
    leaq (%rcx), %r15 // *inFile
L7:
    movq 16(%rsp), %r14   // buf128
    movq 8(%rsp), %r12   // len
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
    vmulps %xmm1, %xmm1, %xmm2
    vpermilps $0xB1, %xmm2, %xmm3
    vaddps %xmm2, %xmm3, %xmm2
    vmulps all_hundredths(%rip), %xmm2, %xmm2
    vcmpps $0x1D, %xmm1, %xmm2, %xmm2
    vandps %xmm2, %xmm1, %xmm1
nosquelch:
    // window filter
    movq %r12, %rcx
//    negq %rcx
//    vpermilps $0x4E, %xmm1, %xmm2
//    vaddps %xmm1, %xmm2, %xmm2
//    shrq $5, %rcx
//    shlq $4, %rcx
//    negq %rcx
    vmovaps %xmm1, (%r14, %rcx)

    addq $16, %r12
    jl L6
    test %r13, %r13
    jz loopEnd

//    movb (%rsp), %cl
//    testb %cl, %cl
//    jz isNotRdc           // if isRdc
//// removeDCSpike(buf128, DEFAULT_BUF_SIZE);
//    movq %r14, %rdi
//    movl $1024, %esi
//    callq _removeDCSpike // TODO change to actual label name once in own .S file
isNotRdc:
// demodulateFmData(buf128, depth, result);
    movq %r14, %rdi
    movl $1024, %esi
    leaq _result(%rip), %rdx
    callq _demodulateFmData // TODO change to actual label
// fwrite(result, OUTPUT_ELEMENT_BYTES, depth, outFile);
    leaq _result(%rip), %rdi
    movl $4, %esi
    movl $1024, %edx
    movq 24(%rsp), %rcx // TODO result put in header.S
    callq _fwrite // TODO change macro-ifdef-defined label
loopEnd: 
    movq _exitFlag(%rip), %rcx
    test %rcx, %rcx
    jz L7

end: 
    addq $40, %rsp
    popq %r15
    popq %r14
    popq %r13
    popq %r12
    popq %rbx
    popq %rbp
    ret