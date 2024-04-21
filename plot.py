#
# This file is part of the demodulator distribution
# (https://github.com/peads/demodulator).
# with code originally part of the misc_snippets distribution
# (https://github.com/peads/misc_snippets).
# Copyright (c) 2023-2024 Patrick Eads.
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
import numpy as np
from scipy import fft

bufsize = 32768
plt.style.use('dark_background')
fig, (ax_f, ax_t) = plt.subplots(2, 1, constrained_layout=True)
ax_f.set_xlim(-0.11, 0.11)
ax_f.set_ylim(-5, 2500)
ax_t.set_ylim(-25, 25)
# figManager = plt.get_current_fig_manager()
# figManager.window.showMaximized()


class Chunker(Iterable):
    def __init__(self, file, N=bufsize, p='f'):
        self.file = file
        self.bufsize = N
        self.chunksize = self.bufsize >> 1
        self.fs = self.bufsize * p
        self.prec = 3 if prec != 'f' else 2

    def __iter__(self):
        return self

    def __next__(self):
        data = list(struct.unpack_from(self.fs, self.file.read(self.bufsize << self.prec)))
        if bool(data):
            return data
        raise StopIteration()


def animate(y, N):
    try:
        f_xlim = ax_f.get_xlim()
        f_ylim = ax_f.get_ylim()

        # w_xlim = ax_w.get_xlim()
        # w_ylim = ax_w.get_ylim()

        t_xlim = ax_t.get_xlim()
        t_ylim = ax_t.get_ylim()

        ax_f.clear()
        # ax_w.clear()
        ax_t.clear()

        ax_f.set_xlim(f_xlim)
        ax_f.set_ylim(f_ylim)

        ax_t.set_xlim(t_xlim)
        ax_t.set_ylim(t_ylim)

        amps = np.abs(fft.fft(y, n=N, norm='backward'))
        freqs = np.fft.fftfreq(len(amps))
        ax_f.plot(freqs, amps)
        # ax_w.specgram(y, Fs=1 / len(y), sides='twosided')
        ax_t.plot(np.arange(0.0, 1.0, 1 / len(y)), y)

        # if w_xlim != (0, 1) and w_ylim != (0, 1):
        #     ax_w.set_xlim(w_xlim)
        #     ax_w.set_ylim(w_ylim)
    except ZeroDivisionError:
        pass
    return fig


argc = len(sys.argv)
sampRate = int(sys.argv[1]) if argc > 1 else 125000
print(f'Sampling rate: {sampRate}')
floatPrecision = 'd' if argc > 2 and int(sys.argv[2]) else 'f'
prec = 3 if floatPrecision != 'f' else 2
shift = int(sys.argv[3]) if argc > 3 else 0
if shift > 0:
    bufsize <<= shift
else:
    bufsize >>= -shift
print(f'Size of buffer: {bufsize}')
fftlen = bufsize >> 1
print(f'FFT len: {fftlen}')

with open(sys.stdin.fileno(), "rb", closefd=False) as f:
    chkr = Chunker(f, bufsize, floatPrecision)
    ani = animation.FuncAnimation(fig,
                                  animate,
                                  fargs=(fftlen,),
                                  frames=iter(chkr),
                                  save_count=8,
                                  interval=40)

    plt.show()
