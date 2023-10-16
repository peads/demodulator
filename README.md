# demodulator
Yet another FM demodulator, but now in three flavors because I like pain! (Intel intrinsics [requires avx2 and/or avx512(bw|dq|f)], Vanilla C99, and CUDA). Basically, this is just a way for me to spend (i.e. waste) my spare time playing with different ways to do DSP.
## Building
Clone this repo, then use cmake to build. The code has been tested on various *nix platforms (e.g. Ubuntu 18+ aarch64, Debian buster armhf, Ubuntu 20+ x64, MacOS 13.3 and Ubuntu on WSL2), but the various automation scripts may not function as expected on all systems.

`mkdir build && cd build && cmake .. && make -j$(nproc)`
#### CMake compile options 
- ~~`-DIS_ASSEMBLY` compiles assembly language versions, if available (x64 and aarch64 only), default: OFF~~ (deprecated in favor of intrinsics, but the code base retains the files for posterity; it is no longer supported)
- `-DIS_NVIDIA` compiles CUDA version, requires `nvcc` (https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) , default: OFF
- `-DNO_INTRINSICS` compiles non-intrinsics-based version, even if extensions are available (x64 only), default: OFF
- `-DNO_AVX512` compiles intrinsics version, but forces avx2 version if avx512 extension are detected on the system, default: OFF
## Usage
The demodulator doesn't care about input sampling rate (depending on which flavor you compiled--the avx2 version decimates the sampling rate 16 times, the avx512 version decimates it by 4, and the C and CUDA versions also quarter it--ymmv), but it does expect the input to be uint8 encoded. Just feed it some raw, 8-bit I/Q data, and with some further pipe work with SoX you have wonderfully useful/-less audio.
### Examples
#### Converting and piping via SoX a pre-recorded file to the demodulator, and decoding the digital audio with DSD (this assumes the avx2 version was compiled):
```
sox -D -twav <some-raw-iq-wav-file> -traw -b8 -eunsigned-integer -r1024k - \
| build/demodulator -i - -o - \
| sox -traw -ef -r64k -b32 - -traw -b16 -es -r48k - \
| dsd -i - -o /dev/null
```
#### Piping raw rtl_sdr data to the demodulator, converting the output with SoX, and decoding with multimon-ng (this assumes any of the other flavors were compiled)
```
rtl_sdr -f<freq> -s1024000 - \
| build/demodulator -i - -o - \
| sox -traw -ef -r256k -b32 - -traw -b16 -r22050 -es - \
| multimon-ng -a SCOPE -a<only-the-choicest-codecs> -
```
