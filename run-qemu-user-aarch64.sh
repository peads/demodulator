#!/bin/bash
 #
 # This file is part of the demodulator distribution
 # (https://github.com/peads/demodulator).
 # with code originally part of the misc_snippets distribution
 # (https://github.com/peads/misc_snippets).
 # Copyright (c) 2023 Patrick Eads.
 #
 # This program is free software: you can redistribute it and/or modify
 # it under the terms of the GNU General Public License as published by
 # the Free Software Foundation, version 3.
 #
 # This program is distributed in the hope that it will be useful, but
 # WITHOUT ANY WARRANTY; without even the implied warranty of
 # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 # General Public License for more details.
 #
 # You should have received a copy of the GNU General Public License
 # along with this program. If not, see <http://www.gnu.org/licenses/>.
 #

rm -rf build/ ||:
mkdir build
# use cflags for a build representative of target that could be an rpi4
cmake -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_MAKE_PROGRAM=ninja \
  -DCMAKE_C_COMPILER=/usr/bin/aarch64-linux-gnu-gcc \
  -DCMAKE_C_FLAGS="-mabi=lp64 -march=armv8-a+crc -mcpu=cortex-a72 -mtune=cortex-a72 -mfix-cortex-a53-835769 -mfix-cortex-a53-843419 -momit-leaf-frame-pointer" \
  -DIS_NATIVE=OFF -DNO_INTRINSICS=ON -DIS_VERBOSE=ON \
  -G Ninja -S . -B build
cmake --build build
sox -v1.75 -q -D -twav SDRSharp_20160101_231914Z_12kHz_IQ.wav  -traw -eunsigned-int -b8 -r384k - \
  | qemu-aarch64 -L /usr/aarch64-linux-gnu/ build/demodulator -i - -o - \
  | sox -q -D -traw -b32 -ef -r192k - -traw -es -b16 -r48k - \
  | dsd -i - -o /dev/null -n
