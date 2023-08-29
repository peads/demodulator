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
#ifndef DEMODULATOR_NVIDIA_CUH
#define DEMODULATOR_NVIDIA_CUH
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <unistd.h>
#include <cstdlib>
#include "definitions.h"
#include "matrix.h"
#define INPUT_ELEMENT_BYTES 1
#define BLOCKDIM 1024
static const int GRIDDIM = (DEFAULT_BUF_SIZE + BLOCKDIM - 1) / BLOCKDIM;
#endif //DEMODULATOR_NVIDIA_CUH
