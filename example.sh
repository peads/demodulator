#!/bin/bash

wavFile=SDRSharp_20160101_231914Z_12kHz_IQ.wav
if [ ! -z "$1" ]; then
  wavFile=$1
fi

audioOutOpts="-o/dev/null -n"
if [ ! -z "$2" ]; then
    audioOutOpts="-o${2}"
fi

hasAvx2=$(cat /proc/cpuinfo | grep avx2 | sed -E 's/avx2/yes/g' | grep yes | wc -l)
hasAvx512=$(cat /proc/cpuinfo | grep avx512 | sed -E 's/avx512(bw|dq|f)/yes/g' | grep yes | wc -l)

declare -A opts=(
  ["-DIS_INTRINSICS=OFF -DIS_NATIVE=ON"]="256k"
)

if [ $hasAvx2 -ge 1 ]; then
  opts["-DIS_INTRINSICS=ON -DIS_NATIVE=ON"]="128k"
fi

if [ $hasAvx512 -ge 4 ]; then
  opts["-DIS_INTRINSICS=ON -DIS_NATIVE=ON"]="128k"
  opts["-DIS_INTRINSICS=ON -DNO_AVX512=ON -DIS_NATIVE=ON"]="128k"
fi

#type nvcc >/dev/null 2>&1
#if [ "$?" == 0 ]; then
#  opts["-DIS_NVIDIA=ON"]="256k"
#fi

declare -A arr

function findCompiler() {

  local __resultvar=$2

  type $1 >/dev/null 2>&1
  result="$?"
  if [ "$result" == 0 ]; then

    echo ":: COMPILER INFO"
    echo "$(${1} --version)"
    echo ":: END COMPILER INFO"
    key="-DCMAKE_C_COMPILER=$(which ${1})"
    for val in "${!opts[@]}"; do
      arr[$key $val]=${opts[$val]}
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

findCompiler gcc hasGcc
findCompiler clang hasClang
findCompiler icc hasIcc
#findCompiler nvcc hasNvcc

set -e

for key in "${!arr[@]}"; do

  val=${arr[$key]}
#  isNvcc=`echo $key | grep "nvcc" | wc -l`
#  isNVIDIA=`echo $key | grep "NVIDIA" | wc -l`
#
#  if [ $isNvcc == 1 ]; then
#    if [ $isNVIDIA != 1 ]; then
#      continue
#    else
#      key="-DIS_NATIVE=ON -DIS_NVIDIA=ON"
#      val="256k"
#    fi
#  fi

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

  echo "Press any key to continue..."
  read -s -n 1
  echo "You pressed a key! Continuing..."
done

