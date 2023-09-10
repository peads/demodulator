#!/bin/bash

sox -D -twav SDRSharp_20160101_231914Z_12kHz_IQ.wav -traw -eunsigned-int -b8 -r512k - | build/demodulator -i - -o -r1 - | sox -traw -b32 -ef -r128k - -traw -es -b16 -r48k - | dsd -i - -o /dev/null -n
sox -D -twav SDRSharp_20160101_231914Z_12kHz_IQ.wav -traw -es -b16 -r512k - | build/demodulator -i - -o - | sox -traw -b32 -ef -r128k - -traw -es -b16 -r48k - | dsd -i - -o /dev/null -n