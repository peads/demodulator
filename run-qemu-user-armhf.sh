#!/bin/bash
 #
 # This file is part of the demodulator distribution
 # (https://github.com/peads/demodulator).
 # with code originally part of the misc_snippets distribution
 # (https://github.com/peads/misc_snippets).
 # Copyright (c) 2023-2024 Patrick Eads.
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
# use cflags for a build representative of target that could be an rpizerow
cmake -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_MAKE_PROGRAM=ninja \
  -DCMAKE_C_COMPILER=/usr/bin/arm-linux-gnueabihf-gcc \
  -DCMAKE_C_FLAGS="-march=armv6zk+fp -mcpu=arm1176jzf-s -marm -mtune=arm1176jz-s -mfloat-abi=hard -mfpu=vfp -mtp=cp15" \
  -DIS_NATIVE=OFF -DNO_INTRINSICS=ON -DIS_VERBOSE=ON \
  -G Ninja -S . -B build
cmake --build build
sox -v2 -q -D -twav SDRSharp_20160101_231914Z_12kHz_IQ.wav  -traw -eunsigned-int -b8 -r192k - \
    | qemu-arm -L /usr/arm-linux-gnueabihf/  build/demodulator -i - -o - -S96000 -l12500 -L12500 \
    | sox -q -D -traw -b32 -ef -r96k - -traw -es -b16 -r48k - \
    | dsd -i - -o /dev/null -n
sox -v50 -q -D -twav FLEX_Pager_IQ_20150816_929613kHz_IQ.wav -traw -b8 -eunsigned-int -r192k -c2 - \
    | qemu-arm -L /usr/arm-linux-gnueabihf/ build/demodulator -i - -o - -S96000 -l6500 -L12500 \
    | sox -q -D -traw -r96k -ef -b32 - -traw -b16 -es -r22050 - 2>/dev/null \
    | multimon-ng -q -c -aFLEX_NEXT -
