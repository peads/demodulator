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
from scipy import signal
import sys
import struct
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from functools import partial
import numpy as np

plt.style.use('dark_background')
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
displaysize = 4096
bufsize = displaysize << 4
dt = displaysize >> 3


class Chunker:
    def __init__(self, file):
        self.file = file
        self.fs = bufsize * 'f'
        self.ymins = dt * 0
        # self.b, self.a = signal.butter(N=8, Wn=0.1, btype='low')

    def __iter__(self):
        try:
            self.chunk = list(struct.unpack_from(self.fs, f.read(bufsize << 2)))
            return self
        except struct.error as ex:
            raise StopIteration(ex)

    def __next__(self):
        if bool(self.chunk):
            result = self.chunk[0:dt]
            del self.chunk[0:dt]
            return self.ymins, result
        raise StopIteration()


def generateData(file):
    chunker = Chunker(file)
    return iter(chunker)


def animate(i, ts, ys):
    # if (bool(ts)):
    #    ts.append(ts[-1] + 1)
    # else:
    #    ts.append(1)
    # ys.append(i)
    (ymins, y) = i
    ts.extend(range(ts[-1], ts[-1] + len(y)))
    ys.extend(y)
    ts = ts[-displaysize:]
    ys = ys[-displaysize:]

    xlim=ax.get_xlim()
    ylim=ax.get_ylim()
    ax.clear()
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)

    fft_data = np.abs(np.fft.fft(ys))
    fft_freq = np.fft.fftfreq(len(fft_data))

    ax.plot(fft_freq, fft_data)
    # ax.plot(ts, ys)
    # ax.scatter(ts, ys)
    # ax.fill_between(ts, ymins, ys, alpha=1, linewidth=dt)


with open(sys.stdin.fileno(), "rb", closefd=False) as f:
    # needs initial value, s.t. animate doesn't whine on extend
    ys = [0]
    ts = [0]
    ani = animation.FuncAnimation(fig, animate, fargs=(ts, ys), frames=partial(generateData, f),
                                  save_count=displaysize, interval=32)
    plt.ylim(0, 1e3)
    plt.xlim(-0.5, 0.5)
    # plt.axis('off')
    plt.show()
