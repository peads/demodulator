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

wavFile=SDRSharp_20160101_231914Z_12kHz_IQ.wav
wavFile2=FLEX_Pager_IQ_20150816_929613kHz_IQ.wav
audioOutOpts=""
compilers=()

if [ ! -z "$1" ] && [ "$1" ~= "(y|Y)" ]; then
  dontWait=$1
fi
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
   time build/demodulator -i uint8.dat -o file
}

function executeRun() {
  sox -v2 -q -D -twav ${wavFile} -traw -eunsigned-int -b8 -r384k - 2>/dev/null \
    | tee -i uint8.dat \
    | build/demodulator -i - -o - -h10 \
    | sox -q -D -traw -b32 -ef -r${2} - -traw -es -b16 -r48k - 2>/dev/null \
    | dsd -i - -o/dev/null -n ${audioOutOpts}
}

function executeRun2() {
  sox -q -D -v20 -twav ${wavFile2} -traw -b8 -eunsigned-int -r250k -c2 - 2>/dev/null \
    | build/demodulator -i - -o - -h10 -l10000 \
    | tee -i uint8.dat \
    | sox -traw -r${1} -ef -b32 - -traw -b16 -es -r22050 - 2>/dev/null \
    | multimon-ng -q -c -aFLEX_NEXT -
}

function join_by() {
  local IFS="$1"; shift; echo "$*";
}

function findSmVersion() {
  nvc=`which nvcc`
  var=(`echo ${nvc} | sed 's/\// /g'`)
  unset var[-1]
  unset var[-1]
  var="/`join_by "/" ${var[*]}`/extras/demo_suite/deviceQuery"
  var=`eval $var | grep "CUDA Capability Major/Minor version number" \
    | sed -e "s/\s\+CUDA Capability Major\/Minor version number:\s\+//g"`
  echo $var
}

findCompiler gcc hasGcc
findCompiler clang hasClang
findCompiler icc hasIcc
findCompiler nvcc hasNvcc
#if [ $hasNvcc == 0 ]; then
#  var=$(bc -l <<<"$(findSmVersion) < 8")
#  if [ $var == 0 ]; then
#    gain=$(bc -l <<<"-1*${gain}")
#  fi
#  echo $gain
#fi
set -e
i=0
for compiler in ${compilers[@]}; do
  ./cmake_build.sh "-DCMAKE_C_COMPILER=${compiler} -DIS_NATIVE=ON" | grep "The C compiler identification"
  executeRun $compiler "192k" 1

  echo ":: STARTING TIMED RUNS 1 FOR: ${compiler} (default options)"
  executeTimedRun
  executeTimedRun
  executeTimedRun
  echo ":: COMPLETED TIMED RUNS 1 FOR: ${compiler} (default options)"
  rm -rf file uint8.dat

  executeRun2 "125k" 1
  echo ":: STARTING TIMED RUNS 2 FOR: ${compiler} (default options)"
  executeTimedRun
  executeTimedRun
  executeTimedRun
  echo ":: COMPLETED TIMED RUNS 2 FOR: ${compiler} (default options)"
  rm -rf file uint8.dat

  ./cmake_build.sh "-DCMAKE_C_COMPILER=${compiler} -DIS_NATIVE=ON -DIS_NVIDIA=OFF -DNO_INTRINSICS=OFF -DNO_AVX512=ON" | grep "The C compiler identification"
  executeRun $compiler "192k" 1

  echo ":: STARTING TIMED RUNS 1 FOR: ${compiler} -DNO_AVX512=ON"
  executeTimedRun
  executeTimedRun
  executeTimedRun
  echo ":: COMPLETED TIMED RUNS 1 FOR: ${compiler} -DNO_AVX512=ON"
  rm -rf file uint8.dat

  executeRun2 "125k" 1
  echo ":: STARTING TIMED RUNS 2 FOR: ${compiler} -DNO_AVX512=ON"
  executeTimedRun
  executeTimedRun
  executeTimedRun
  echo ":: COMPLETED TIMED RUNS 2 FOR: ${compiler} -DNO_AVX512=ON"
  rm -rf file uint8.dat

  ./cmake_build.sh "-DCMAKE_C_COMPILER=${compiler} -DIS_NATIVE=ON -DIS_NVIDIA=ON" | grep "The C compiler identification"
  executeRun $compiler "96k" 1

  echo ":: STARTING TIMED RUNS FOR: ${compiler} -DIS_NVIDIA=ON"
  executeTimedRun
  executeTimedRun
  executeTimedRun
  echo ":: COMPLETED TIMED RUNS FOR: ${compiler} -DIS_NVIDIA=ON"
  rm -rf file uint8.dat

  ./cmake_build.sh "-DCMAKE_C_COMPILER=${compiler} -DIS_NATIVE=ON -DIS_NVIDIA=OFF -DNO_INTRINSICS=ON -DNO_AVX512=OFF" | grep "The C compiler identification"
  executeRun $compiler "192k" 1

  echo ":: STARTING TIMED RUNS 1 FOR: ${compiler} -DNO_INTRINSICS=ON"
  executeTimedRun
  executeTimedRun
  executeTimedRun
  echo ":: COMPLETED TIMED RUNS 1 FOR: ${compiler} -DNO_INTRINSICS=ON"
  rm -rf file uint8.dat

  executeRun2 "125k" 1
  echo ":: STARTING TIMED RUNS 2 FOR: ${compiler} -DNO_INTRINSICS=ON"
  executeTimedRun
  executeTimedRun
  executeTimedRun
  echo ":: COMPLETED TIMED RUNS 2 FOR: ${compiler} -DNO_INTRINSICS=ON"
  rm -rf file uint8.dat
  i=$(( i + 1 ))
#  waitForUserIntput $i
done


echo "Job's done."
