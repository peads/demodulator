#
# This file is part of the demodulator distribution
# (https://github.com/peads/demodulator).
# with code originally part of the misc_snippets distribution
# (https://github.com/peads/misc_snippets).
# Copyright (c) 2023 Patrick Eads.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
import sys
import struct
from typing import Iterable
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from functools import partial
import numpy as np
from scipy import fft

plt.style.use('dark_background')
fig, (ax_t, ax_w, ax_a) = plt.subplots(3, 1, constrained_layout=True)
ax_t.set_xlim(-0.51, 0.51)
ax_t.set_ylim(0, 100)
ax_a.set_ylim(-1.5, 1.5)


class Chunker(Iterable):
    def __init__(self, file, bufsize=250000):
        self.file = file
        self.bufsize = bufsize
        self.chunksize = self.bufsize >> 1
        self.fs = self.bufsize * 'f'
        self.chunk = None
        # self.b, self.a = signal.butter(N=8, Wn=0.1, btype='low')

    def __iter__(self):
        data = f.read(self.bufsize << 2)
        if bool(data):
            self.chunk = list(struct.unpack_from(self.fs, data))
        return self

    def __next__(self):
        if bool(self.chunk):
            result = self.chunk[0:self.chunksize]
            del self.chunk[0:self.chunksize]
            return result
        raise StopIteration()


def generateData(file, bufsize):
    try:
        chunker = Chunker(file, bufsize)
        return iter(chunker)
    except TypeError:
        pass


def animate(y, dt, fftlen):
    t_xlim = ax_t.get_xlim()
    t_ylim = ax_t.get_ylim()

    w_xlim = ax_w.get_xlim()
    w_ylim = ax_w.get_ylim()

    a_xlim = ax_a.get_xlim()
    a_ylim = ax_a.get_ylim()

    ax_t.clear()
    ax_w.clear()
    ax_a.clear()

    ax_t.set_xlim(t_xlim)
    ax_t.set_ylim(t_ylim)

    ax_a.set_xlim(a_xlim)
    ax_a.set_ylim(a_ylim)

    # amps = np.abs(np.fft.fft(y))
    # freqs = np.fft.fftfreq(len(fft_data))
    amps = np.abs(fft.fft(y, n=fftlen, norm='backward'))
    freqs = np.fft.fftfreq(len(amps))
    ax_t.plot(freqs, amps)
    ax_w.specgram(y, Fs=1/len(y))#, sides='twosided')
    ax_a.plot(np.arange(0.0, 1.0, 1/len(y)), y)

    if w_xlim != (0, 1) and w_ylim != (0, 1):
        ax_w.set_xlim(w_xlim)
        ax_w.set_ylim(w_ylim)

    return fig


argc = len(sys.argv)
sampRate = int(sys.argv[1]) if argc > 1 else 125000
print(f'Sampling rate: {sampRate}')
bufsize = 32768
shift = int(sys.argv[2]) if argc > 2 else 0
if shift > 0:
    bufsize <<= shift
else:
    bufsize >>= -shift
print(f'Size of buffer: {bufsize}')
fftlen = bufsize >> 1 #pow(2, int(np.ceil(np.log2(max(bufsize, sampRate)))))
print(f'FFT len: {fftlen}')

with open(sys.stdin.fileno(), "rb", closefd=False) as f:
    dt = 1 / sampRate
    ani = animation.FuncAnimation(fig, animate, fargs=(dt, fftlen),
                                  frames=partial(generateData, f, bufsize),
                                  save_count=8, interval=40)
    plt.show()
