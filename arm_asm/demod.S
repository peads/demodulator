#include "asm.h"
.data

RIGHT:
//    .hword 0x0045
    .word 0xbf147ae1//0x3ed70a3d

LEFT:
    .byte 0x46

MSG:
    .asciz "Nice.\n"
LEN = . - MSG

    .text
    str     x30, [sp, #-16]!    // store the link reg
    str     x5, [sp, #-16]!     // store callee-preserved regs
 
    mov     x8, #SYS_OPEN
    mov     w0, #AT_FDCWD
    mov     x1, x2              // fd
    mov     w2, #OPEN_FLAGS     // O_TRUNC | O_CREAT | O_WRONLY
    mov     w3, #OPEN_MODE      // rw-rw-r--
    svc     #0
    mov     x5, x0
    cmp     x0, #0              // if unable to open file end
    b.le    end
    
    ldr     x0, =RIGHT
    ldr     s0, [x0]
    ldr     x1, =LEFT
    ldrb    w1, [x1]
    bl      coolNumber

    mov     w8, #SYS_WRITE    
    str     s0, [sp, #1]
    add     x1, sp, #0x1

    mov     x0, x5         // fd
    mov     w2, #4
    //ldr     x1, =MSG
    //ldr     x2, =LEN          // size in bytes
    svc     #0
end:
    ldr     x5, [sp], #16   // reload preserved regs
    ldr     x30, [sp], #16  // reload lr
    ret

    .global coolNumber
coolNumber:
    //leaf functions do not need to push the lr

    //and     w2, w2, #0xFFFF
    //uxth    w2, w2
    //ucvtf   s2, w2

    //and     w3, w3, #0xFF
    sxtb    w1, w1
    ucvtf   s1, w1

    fadd    s0, s1, s0

    ret

