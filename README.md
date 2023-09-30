# demodulator
Yet another FM demodulator, but now in three flavors because I like pain! (x64 asm, C, and CUDA)
## Building
Clone this repo, then use cmake to build.

`mkdir build && cd build && cmake .. && make -j$(nproc)`
#### CMake compile options 
- `-DIS_ASSEMBLY` compiles assembly language versions, if available (x64 and aarch64 only), default: OFF
- `-DIS_NVIDIA` compiles CUDA version, requires `nvcc`, default: OFF
- `-DIS_INTRINSICS` compiles intrinsics version, if available (x64 only), default: OFF
## Usage
The demodulator doesn't care about sampling rate; feed it some raw I/Q data, but depending on which flavor you compiled (the asm version does not affect sampling rate, the C and CUDA versions halve it) ymmv, and with some further pipe work with SoX you have wonderfully useful/-less audio.
 #### Converting and piping via SoX a pre-recorded file to the demodulator, and decoding the digital audio with DSD.
```
sox -twav <some-raw-iq-wav-file> -traw -b8 -eunsigned-integer -r512k - \
| build/demodulator -r1 -i - -o - \
| sox -traw -ef -r256k -b32 - -traw -b16 -es -r48k - \
| dsd -i - -o /dev/null
```
#### Piping raw rtl_sdr data to the demodulator, converting the output with SoX, and decoding with multimon-ng
```
rtl_sdr -f<freq> -s1024000 - \
| build/demodulator -r1 -i - -o - \
| sox -traw -ef -r512k -b32 - -traw -b16 -r22050 -es - \
| multimon-ng -a SCOPE -a<only-the-choicest-codecs> -
```
