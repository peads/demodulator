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
from scipy import signal as sig
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from functools import partial

import numpy as np

plt.style.use('dark_background')
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
fs = 250000
dt = 1 / fs
bufsize = 262144 #int(pow(2, np.floor(np.log2(fs))))


class Chunker(Iterable):
    def __init__(self, file):
        self.file = file
        self.fs = bufsize * 'f'
        self.chunk = None
        # self.b, self.a = signal.butter(N=8, Wn=0.1, btype='low')

    def __iter__(self):
        data = f.read(bufsize << 2)
        if bool(data):
            self.chunk = list(struct.unpack_from(self.fs, data))
        return self

    def __next__(self):
        if bool(self.chunk):
            chunksize = bufsize >> 1
            result = self.chunk[0:chunksize]
            del self.chunk[0:chunksize]
            return result
        raise StopIteration()


def generateData(file):
    try:
        chunker = Chunker(file)
        return iter(chunker)
    except TypeError:
        pass


def animate(y):

    ax.clear()
    # fft_data = np.abs(np.fft.fft(ys))
    # fft_freq = np.fft.fftfreq(len(fft_data))
    # ax.plot(fft_freq, fft_data)
    return ax.specgram(y, Fs=dt, Fc=0, NFFT=1024, window=sig.get_window('nuttall',1024))


with open(sys.stdin.fileno(), "rb", closefd=False) as f:

    ani = animation.FuncAnimation(fig, animate, frames=partial(generateData, f), #partial(generateData, f),
                                  save_count=8, interval=1)
    plt.show()
