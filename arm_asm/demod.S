#include "asm.h"
    .data
    /* Data segment: define our message string and calculate its length. */
msg:
    .ascii        "Hello, ARM64!\n"
    len = . - msg

    .text

/* Our application's entry point. */

    /* syscall write(int fd, const void *buf, size_t count) */
    mov     x0, #1      /* fd := STDOUT_FILENO */
    ldr     x1, =msg    /* buf := msg */
    ldr     x2, =len    /* count := len */
    mov     w8, #64     /* write is syscall #64 */
    svc     #0          /* invoke syscall */

    mov     x0, #-1     // EOF in linux-abi
    ret

