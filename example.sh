#!/bin/bash

#sox -G -twav SDRSharp_20160101_231914Z_12kHz_IQ.wav -traw -b8 -eunsigned-integer -r512k - | build/demodulator -i - -o - | sox -G -traw -ef -r256k -b32 - -traw -b16 -es -r48k - | dsd -i - -o /dev/null -n -w uint8.wav
#sox --norm=-3 -twav SDRSharp_20160101_231914Z_12kHz_IQ.wav -traw -b16 -es -r512k - | build/demodulator -i - -o - | sox  -traw -ef -r256k -b32 - -traw -b16 -es -r48k - | dsd -i - -o /dev/null -n int16.wav
#sox --norm=-3 -twav SDRSharp_20160101_231914Z_12kHz_IQ.wav -traw -b8 -eunsigned-integer -r512k - | build/demodulator -i - -o - | sox -traw -ef -r512k -b32 - -traw -b16 -es -r48k - | dsd -i - -o /dev/null -n uint8_asm.wav
sox -twav SDRSharp_20160101_231914Z_12kHz_IQ.wav -traw -eunsigned-int -b8 -r512k - | build/demodulator -i - -o - | sox -traw -b32 -ef -r128k - -traw -es -b16 -r48k - | dsd -i - -o /dev/null -n