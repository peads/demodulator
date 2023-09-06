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
#ifndef DEMODULATOR_ASM_H
#define DEMODULATOR_ASM_H

#include "definitions.h"
                        // S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH
#define OPEN_MODE       0x1b4
#define STACK_SIZE      DEFAULT_BUF_SIZE
#ifdef __APPLE__
    #define FREAD           _fread
    #define STATUS          _checkFileStatus
    #define FERROR          _ferror
    #define FEOF            _feof
    #define PERROR          _perror
    //DARWIN x64 ONLY
    #define SYS_WRITE       0x2000004
    #define SYS_OPEN        0x2000005
    #define SYS_CLOSE       0x2000006
                            // O_TRUNC | O_CREAT | O_WRONLY
    #define OPEN_FLAGS      0x601

    .globl  _processMatrix
    _processMatrix:
#else
    #if defined(__aarch64__) || defined(_M_ARM64)
        #define SYS_WRITE   0x40
        // this is actually openat(3), but whatever 
        #define SYS_OPEN    0x38
        #define SYS_CLOSE   0x39
        #define AT_FDCWD    0xffffff9c
    #else 
        #define SYS_WRITE   0x1
        #define SYS_OPEN    0x2
        #define SYS_CLOSE   0x3
    #endif
                            // O_TRUNC | O_CREAT | O_WRONLY
    #define OPEN_FLAGS      0x241
    #define FREAD           fread
    #define STATUS          checkFileStatus
    #define FERROR          ferror
    #define FEOF            feof
    #define PERROR          perror

    .globl  processMatrix
    processMatrix:
#endif
#endif //DEMODULATOR_ASM_H
