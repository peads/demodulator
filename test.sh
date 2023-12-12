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
PREC="-DSET_PRECISION=ON"
wavFile=SDRSharp_20160101_231914Z_12kHz_IQ.wav
wavFile2=FLEX_Pager_IQ_20150816_929613kHz_IQ.wav
audioOutOpts=""
compilers=()

if [ ! -z "$2" ]; then
  wavFile=$2
fi
if [ ! -z "$3" ]; then
  audioOutOpts="-w$3"
fi

function findCompiler() {

  local __resultvar=$2

  type $1 >/dev/null 2>&1
  result="$?"
  if [ "$result" == 0 ]; then

    echo ":: COMPILER INFO"
    echo "$(${1} --version)"
    echo ":: END COMPILER INFO"
    if [ "${1}" != nvcc ]; then
      compilers+=("`which ${1}`")
    fi
  fi

  eval $__resultvar="'$result'"
}

function executeTimedRun() {
   time build/demodulator -i uint8.dat -o file -S96000 -l6500 ${1}
}

function executeRun() {
  sox -v2 -q -D -twav ${wavFile} -traw -eunsigned-int -b8 -r192k - 2>/dev/null \
    | tee -i uint8.dat \
    | build/demodulator -i - -o - -w2 -d5 -m3 -l12500 -S96000 ${1} \
    | sox -v0.5 -q -D -traw -b64 -ef -r96k - -traw -es -b16 -r48k - 2>/dev/null \
    | dsd -i - -o/dev/null -n
}

function executeRun2() {
  sox -v100 -q -D -twav  ${wavFile2} -traw -eunsigned-int -b8 -r192k - 2>/dev/null \
    | tee -i uint8.dat     \
    | build/demodulator -w1 -d7 -i - -o - -l9600 -S96000 ${1} \
    | sox -v0.5 -q -D -traw -b64 -ef -r96k - -traw -es -r22050 -b16 - 2>/dev/null \
    | multimon-ng -q -c -aFLEX_NEXT -i -
}

findCompiler gcc hasGcc
findCompiler clang hasClang
findCompiler icc hasIcc
findCompiler nvcc hasNvcc

set -e
i=0
for compiler in ${compilers[@]}; do
  ./cmake_build.sh "${PREC} -DCMAKE_C_COMPILER=${compiler} -DIS_NATIVE=ON" | grep "The C compiler identification"
  executeRun

  echo ":: STARTING TIMED RUNS 1 FOR: ${compiler} dsd no lowpass in"
  executeTimedRun
  executeTimedRun
  executeTimedRun
  echo ":: COMPLETED TIMED RUNS 1 FOR: ${compiler} dsd no lowpass in"
  rm -rf file uint8.dat

  executeRun2
  echo ":: STARTING TIMED RUNS 2 FOR: ${compiler} multimon-ng no lowpass in"
  executeTimedRun
  executeTimedRun
  executeTimedRun
  echo ":: COMPLETED TIMED RUNS 2 FOR: ${compiler} multimon-ng no lowpass in"
  rm -rf file uint8.dat

  executeRun "-D7 -L12500"

  echo ":: STARTING TIMED RUNS 1 FOR: ${compiler} dsd with lowpass in"
#  executeTimedRun "-L12500 -w2 -m2"
  executeTimedRun "-D7 -L12500"
  executeTimedRun "-D7 -L12500"
  executeTimedRun "-D7 -L12500"
  echo ":: COMPLETED TIMED RUNS 1 FOR: ${compiler} dsd with lowpass in"
  rm -rf file uint8.dat

  executeRun2 "-m3 -L96000"

  echo ":: STARTING TIMED RUNS 2 FOR: ${compiler} multimon-ng with lowpass in"
  executeTimedRun "-m3 -L96000"
  executeTimedRun "-m3 -L96000"
  executeTimedRun "-m3 -L96000"
  echo ":: COMPLETED TIMED RUNS 2 FOR: ${compiler} multimon-ng with lowpass in"
  rm -rf file uint8.dat
done
echo "Job's done."
