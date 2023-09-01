#include "asm.h"
.data

RIGHT:
//    .hword 0x0045
    .word 0xbf147ae1//0x3ed70a3d

LEFT:
    .byte 0x46

//MSG:
//    .ascii "Nice.\n"
//len = . - MSG

.text
    ldr     x2, =RIGHT
    ldr     s2, [x2]
    ldr     x3, =LEFT
    ldrb    w3, [x3]
    bl      coolNumber

    fcvtau  w0,s0
    mov     w8, #0x5D
    svc     #0

    .global coolNumber
coolNumber:
    //leaf functions do not need to push the lr
    //str     x30, [sp, #-16]! // store the link reg

    //and     w2, w2, #0xFFFF
    //uxth    w2, w2
    //ucvtf   s2, w2

    //and     w3, w3, #0xFF
    sxtb    w3, w3
    ucvtf   s3, w3

    fadd    s0, s2, s3

    ret

    .global _start

