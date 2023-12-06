# demodulator
Yet another FM demodulator as a way for me to spend (i.e. waste) my spare time playing with different ways to learn DSP.
## Building
Clone this repo, then use cmake to build. The code has been tested on various *nix platforms (e.g. Ubuntu 18+ aarch64, Debian buster armhf, Ubuntu 20+ x64, MacOS 13.3 and Ubuntu on WSL2), but the various automation scripts may not function as expected on all systems.

`mkdir build && cd build && cmake .. && make -j$(nproc)`
#### CMake compile options 
- `-DIS_NVIDIA` compiles CUDA version, requires `nvcc` (https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) , default: OFF
## Usage
### Required command-line parameters
- `-i` input file; use '-' to specify stdin
- `-o` output file; use '-' to specify stdout
### Optional command-line parameters
- `-L` input signal lowpass cutoff frequency (ωc), default: none, skips input lowpass entirely
- `-l` output signal lowpass cutoff frequency (ωc), default: 1
- `-S` output sampling rate, default: 10
### Examples
#### Using DSD
```
sox -q -D -twav <some-wav-iq-wav-file> -traw -eunsigned-int -b8 -r192k - \
| build/demodulator -i - -o - -S96000 -l12500 \
| sox -q -D -traw -b32 -ef -r96k - -traw -es -b16 -r48k - \
| dsd -i - -o/dev/null -n
```
#### Using multimon-ng
```
sox -q -D -twav <some-wav-iq-wav-file> -traw -eunsigned-int -b8 -r192k - \
| build/demodulator -i - -o - -S96000 -l6500 -L12500 \
| sox -q -D -traw -b32 -ef -r96k - -traw -es -b16 -r22050 - \
| multimon-ng -q -c -a<some-codecs> -
```
