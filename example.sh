#!/bin/bash

arr=()

type gcc >/dev/null 2>&1
if [ "$?" == 0 ]; then
  echo ":: COMPILER INFO"
  echo "`gcc --version`"
  arr[${#arr[@]}]=$(which gcc)
  echo ":: END COMPILER INFO"
fi

type clang >/dev/null 2>&1
if [ "$?" == 0 ]; then
  echo ":: COMPILER INFO"
  echo "`clang --version`"
  arr[${#arr[@]}]=$(which clang)
  echo ":: END COMPILER INFO"
fi

type icc >/dev/null 2>&1
if [ "$?" == 0 ]; then
  echo ":: COMPILER INFO"
  echo "`icc --version`"
  arr[${#arr[@]}]=$(which icc)
  echo ":: END COMPILER INFO"
fi

set -e

for compiler in ${arr[@]}; do

  ./cmake_build.sh "-DIS_VERBOSE=ON -DIS_ASSEMBLY=ON -DCMAKE_C_COMPILER=$compiler"

  sox -D -twav SDRSharp_20160101_231914Z_12kHz_IQ.wav -traw -eunsigned-int -b8 -r512k - | build/demodulator -i - -o - -r1 | sox -traw -b32 -ef -r256k - -traw -es -b16 -r48k - | dsd -i - -o /dev/null -n

  sox -D -twav SDRSharp_20160101_231914Z_12kHz_IQ.wav -traw -es -b16 -r512k - | build/demodulator -i - -o - | sox -traw -b32 -ef -r256k - -traw -es -b16 -r48k - | dsd -i - -o /dev/null -n

  time build/demodulator -i uint8.dat -o file -r1
  time build/demodulator -i uint8.dat -o file -r1
  time build/demodulator -i uint8.dat -o file -r1
  sox -traw -b32 -ef -r256k file -traw -es -b16 -r48k - | dsd -i - -o /dev/null -n
  rm file
  time build/demodulator -i int16.dat -o file
  time build/demodulator -i int16.dat -o file
  time build/demodulator -i int16.dat -o file
  sox -traw -b32 -ef -r256k file -traw -es -b16 -r48k - | dsd -i - -o /dev/null -n
done
