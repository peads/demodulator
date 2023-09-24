audioOutOpts="-o/dev/null -n"
if [ ! -z "$1" ]; then
    audioOutOpts="-o${1}"
fi

declare -A opts=(
  ["-DIS_INTRINSICS=ON"]="128k"
  ["-DIS_INTRINSICS=ON -DNO_AVX512=ON"]="128k"
  ["-DIS_INTRINSICS=OFF"]="256k"
)
declare -A arr

function gen() {

  type $1 >/dev/null 2>&1

  if [ "$?" == 0 ]; then

    echo ":: COMPILER INFO"
    echo "$(${1} --version)"
    echo ":: END COMPILER INFO"
    key="-DCMAKE_C_COMPILER=$(which ${1})"
    for val in "${!opts[@]}"; do
      arr[$key $val]=${opts[$val]}
    done
  fi
}

function printRunInfo() {

  echo ""
  echo ":: RUN INFO"
  echo ":: $2"
  echo ":: COMPILER OPTIONS=$1"
  echo ":: END RUN INFO"
}

gen gcc
gen clang
gen icc

set -e

for key in "${!arr[@]}"; do

  rm -rf file int16.dat uint8.dat ||:
  compiler=`sh -c "./cmake_build.sh \"${key}\" | grep \"The C compiler identification\""`

  printRunInfo "$key" "$compiler"
  sox -D -twav SDRSharp_20160101_231914Z_12kHz_IQ.wav -traw -eunsigned-int -b8 -r512k - | tee -i uint8.dat | build/demodulator -i - -o - -r1 | sox -traw -b32 -ef -r${arr[$key]} - -traw -es -b16 -r48k - | dsd -i - ${audioOutOpts}
  printRunInfo "$key" "$compiler"
  sox -D -twav SDRSharp_20160101_231914Z_12kHz_IQ.wav -traw -es -b16 -r512k - | tee -i int16.dat | build/demodulator -i - -o - | sox -traw -b32 -ef -r${arr[$key]} - -traw -es -b16 -r48k - | dsd -i - ${audioOutOpts}

  time build/demodulator -i uint8.dat -o file -r1 && rm file
  time build/demodulator -i uint8.dat -o file -r1 && rm file
  time build/demodulator -i uint8.dat -o file -r1
  sox -traw -b32 -ef -r${arr[$key]} file -traw -es -b16 -r48k - | dsd -i - -o /dev/null -n && rm file uint8.dat
  time build/demodulator -i int16.dat -o file && rm file
  time build/demodulator -i int16.dat -o file && rm file
  time build/demodulator -i int16.dat -o file
  sox -traw -b32 -ef -r${arr[$key]} file -traw -es -b16 -r48k - | dsd -i - -o /dev/null -n && rm file int16.dat
done

