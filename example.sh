#!/bin/bash

sox -D -twav SDRSharp_20160101_231914Z_12kHz_IQ.wav -traw -b8 -eunsigned-integer -r512k - | build/demodulator -i - -o - | sox -D -v1.25 -traw -ef -r512k -b32 - -traw -b16 -es -r48k - | dsd -i - -o /dev/null

