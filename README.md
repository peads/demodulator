# demodulator
Yet another FM demodulator, but now in three flavors because I like pain! (x64 asm, C, and CUDA)
## Usage
The demodulator doesn't care about sampling rate; feed it some raw I/Q data, but depending on which flavor you compiled (the asm version does not affect sampling rate, the C and CUDA versions halve it) ymmv, and with some further pipe work with sox you have wonderfully useful audio.
 
`sox -twav <some-16-bit-48khz-wav-file> -traw -b8 -eunsigned-integer -r512k - | build/demodulator -i - -o - | sox -traw -ef -r512k -b32 - -traw -b16 -es -r48k - | dsd -i - -o /dev/null`
