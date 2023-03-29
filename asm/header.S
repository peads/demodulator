    .data
    .align 4
    .globl all_hundredths, all_nonetwentysevens, all_64s, all_23s, all_41s, cnj_transform, dc_avg_iq, dc_raw_const, negate_b_im, result
all_hundredths:
    .rept 4
    .single 0.01
    .endr
all_nonetwentysevens:
    .rept 2
    .quad -0x7f7f7f7f7f7f7f7f
    .endr
all_64s:
    .rept 8
    .single 64.0
    .endr
all_23s:
    .rept 8
    .single 23.0
    .endr
all_41s:
    .rept 8
    .single 41.0
    .endr
cnj_transform: .single 1.0, -1.0, 1.0, -1.0
dc_avg_iq: .zero 16
dc_raw_const:
    .rept 4
    .single 1e-05
    .endr
negate_b_im: .single 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0
result:
    .rept 1024
    .quad 0
    .endr