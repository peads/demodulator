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
 # REQUIRES bc (`sudo apt install bc -y` on Debian-based systems)
 # USAGE ./multimon-example.sh 192.168.0.100 5678 250 123.45MHz
 # This expects a tcp server providing raw I/Q data (e.g. rtl_tcp).
 # The RATE value is the sampling rate of the *input* stream in kHz;
 # *not* the expected output of the demodulator (i.e. given the input rate
 # [of the I/Q stream]: 250k, the output rate [of the demodulator] is 62.5k).
 # The units are omitted because I'm lazy. If you're using the vanilla, or CUDA
 # versions, double the input rate because the decimation is only half as much.

ADDR=127.0.0.1
PORT=1234
RATE=256
LABEL=none

if [ ! -z "$1" ]; then
  ADDR="$1"
fi
if [ ! -z "$2" ]; then
  PORT="$2"
fi
if [ ! -z "$3" ]; then
  RATE=$(echo "scale=1; ${3} / 4" | bc)k
fi
if [ ! -z "$4" ]; then
  LABEL="--label $4"
fi

nc $ADDR $PORT | \
  demodulator -r1 -g0.1 -i - -o - | \
    sox -traw -b32 -ef -r$RATE - -traw -es -b16 -r22050 - | \
      multimon-ng -sZVEI1 -sZVEI2 -sZVEI3 -sDZVEI -sPZVEI -sEEA -sEIA -sCCIR \
        -sDUMPCSV -sUFSK1200 -sDTMF -sMORSE_CW -sFLEX -sX10 -sEAS --timestamp \
        $LABEL -i -