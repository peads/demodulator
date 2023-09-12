#!/bin/bash

GCC=$(which gcc)
CLANG=$(which clang)
arr=()

if ! command -v $GCC &>/dev/null; then
  echo "gcc could not be found"
else
  arr[${#arr[@]}]=$GCC
fi

if ! command -v $CLANG &>/dev/null; then
  echo "clang could not be found"
else
  arr[${#arr[@]}]=$CLANG
fi

for compiler in ${arr[@]}; do

  ./cmake_build.sh "-DIS_INTINT=ON -DCMAKE_C_COMPILER=$compiler"

  sox -D -twav SDRSharp_20160101_231914Z_12kHz_IQ.wav -traw -eunsigned-int -b8 -r512k - | build/demodulator -i - -o - -r1 | sox -traw -b32 -ef -r128k - -traw -es -b16 -r48k - | dsd -i - -o /dev/null -n

  sox -D -twav SDRSharp_20160101_231914Z_12kHz_IQ.wav -traw -es -b16 -r512k - | build/demodulator -i - -o - | sox -traw -b32 -ef -r256k - -traw -es -b16 -r48k - | dsd -i - -o /dev/null -n

  time build/demodulator -i uint8.dat -o file -r1
  time build/demodulator -i uint8.dat -o file -r1
  time build/demodulator -i uint8.dat -o file -r1
  sox -traw -b32 -ef -r128k file -traw -es -b16 -r48k - | dsd -i - -o /dev/null -n
  rm file
  time build/demodulator -i int16.dat -o file
  time build/demodulator -i int16.dat -o file
  time build/demodulator -i int16.dat -o file
  sox -traw -b32 -ef -r256k file -traw -es -b16 -r48k - | dsd -i - -o /dev/null -n
done
