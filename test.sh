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
audioOutOpts="-o/dev/null -n"

if [ ! -z "$1" ]; then
  dontWait=1
fi
if [ ! -z "$2" ]; then
  wavFile=$2
fi
if [ ! -z "$3" ]; then
  audioOutOpts="-o${3}"
fi
if [ ! -z "$4" ]; then
  invertOpt=$4
  type nvcc >/dev/null 2>&1
  hasCuda="$?"
fi

hasAvx2=$(cat /proc/cpuinfo | grep avx2 | sed -E 's/avx2/yes/g' | grep yes | wc -l)
hasAvx512=$(cat /proc/cpuinfo | grep avx512 | sed -E 's/avx512(bw|dq|f)/yes/g' | grep yes | wc -l)

declare -A temp=(
  ["-DIS_INTRINSICS=OFF -DIS_NATIVE=ON"]="256k"
)

if [ $hasAvx2 -ge 1 ]; then
  temp["-DIS_INTRINSICS=ON -DIS_NATIVE=ON"]="128k"
fi

if [ $hasAvx512 -ge 4 ]; then
  temp["-DIS_INTRINSICS=ON -DIS_NATIVE=ON"]="128k"
  temp["-DIS_INTRINSICS=ON -DNO_AVX512=ON -DIS_NATIVE=ON"]="128k"
fi

declare -A opts
compilers=()

function findCompiler() {

  local __resultvar=$2

  type $1 >/dev/null 2>&1
  result="$?"
  if [ "$result" == 0 ]; then

    echo ":: COMPILER INFO"
    echo "$(${1} --version)"
    echo ":: END COMPILER INFO"
    compilers+=($1)
    key="-DCMAKE_C_COMPILER=$(which ${1})"
    for val in "${!temp[@]}"; do
      opts[$key $val]=${temp[$val]}
    done
  fi

  eval $__resultvar="'$result'"
}

function printRunInfo() {

  echo ":: RUN INFO"
  echo ":: $2"
  echo ":: COMPILER OPTIONS=$1"
  echo ":: END RUN INFO"
}

function waitForUserIntput() {
  if [ -z "$dontWait" ] && [ "${1}" -lt "${#opts[@]}" ]; then
    echo "Press any key to continue onto next test..."
    read -s -n 1
    echo "Yes, m'lord."
  fi
}

findCompiler gcc hasGcc
findCompiler clang hasClang
findCompiler icc hasIcc

set -e[ "$1" -lt "${#compilers[@]}" ]

if [ "$hasCuda" == 0 ]; then
  runOpts="-DIS_NVIDIA=ON"
  for curr in "${compilers[@]}"; do
    val="256k"
    compiler=`sh -c "./cmake_build.sh \"${runOpts} -DCMAKE_C_COMPILER=${curr}\" | grep \"The C compiler identification\""`

    printRunInfo "${runOpts} -DCMAKE_C_COMPILER=${curr}" "$compiler"
    sox -q -D -twav "${wavFile}" -traw -eunsigned-int -b8 -r512k - 2>/dev/null | tee -i uint8.dat | build/demodulator -i - -o - -r"${invertOpt}" | sox -traw -b32 -ef -r$val - -traw -es -b16 -r48k - | dsd -i - ${audioOutOpts}

    echo ""
    echo ":: Timing uint8"
    printRunInfo "${runOpts} ${curr}" "$compiler"
    time build/demodulator -i uint8.dat -o file -r1
    rm file
    time build/demodulator -i uint8.dat -o file -r1
    rm file
    time build/demodulator -i uint8.dat -o file -r1
    echo ":: End Timing uint8"
    rm -rf file int16.dat uint8.dat ||:

    waitForUserIntput
  done
fi

i=0
for key in "${!opts[@]}"; do

  val=${opts[$key]}

  compiler=`sh -c "./cmake_build.sh \"${key}\" | grep \"The C compiler identification\""`

  printRunInfo "$key" "$compiler"
  sox -q -D -twav "${wavFile}" -traw -eunsigned-int -b8 -r512k - 2>/dev/null | tee -i uint8.dat | build/demodulator -i - -o - -r1 | sox -traw -b32 -ef -r$val - -traw -es -b16 -r48k - | dsd -i - ${audioOutOpts} #>/dev/null 2>&1

  printRunInfo "$key" "$compiler"
  sox -q -D -twav "${wavFile}" -traw -es -b16 -r512k - 2>/dev/null | tee -i int16.dat | build/demodulator -i - -o - | sox -traw -b32 -ef -r$val - -traw -es -b16 -r48k - | dsd -i - ${audioOutOpts} #>/dev/null 2>&1

  echo ""
  echo ":: Timing uint8"
  printRunInfo "$key" "$compiler"
  time build/demodulator -i uint8.dat -o file -r1
  rm file
  time build/demodulator -i uint8.dat -o file -r1
  rm file
  time build/demodulator -i uint8.dat -o file -r1
  #sox -traw -b32 -ef -r$val file -traw -es -b16 -r48k - | dsd -q -i - -o /dev/null -n && rm -f file uint8.dat
  echo ":: End Timing uint8"
  echo ""
  echo ":: Timing int16"
  printRunInfo "$key" "$compiler"
  time build/demodulator -i int16.dat -o file
  rm file
  time build/demodulator -i int16.dat -o file
  rm file
  time build/demodulator -i int16.dat -o file
  #sox -traw -b32 -ef -r$val file -traw -es -b16 -r48k - | dsd -q -i - -o /dev/null -n && rm -f file int16.dat
  echo ":: End Timing int16"
  rm -rf file int16.dat uint8.dat ||:

  i=$(( i + 1 ))
  waitForUserIntput $i
done

echo "Job's done."
