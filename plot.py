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
# WITHOUT ANY WARRANTY without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
import sys
import struct
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from functools import partial
from math import e

# from collections import deque

plt.style.use('dark_background')
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
displaysize = 32768
bufsize = displaysize << 3
dt = displaysize >> 3
scaling = pow(2, -e)


class Chunker:
    def __init__(self, file):
        self.file = file
        self.fs = bufsize * 'f'
        self.ymins = dt * 0

    def __iter__(self):
        self.chunk = list(struct.unpack_from(self.fs, self.file.read(bufsize << 2)))
        return self

    def __next__(self):
        if (bool(self.chunk)):
            result = self.chunk[0:dt]
            del self.chunk[0:dt]
            return self.ymins, result
            # return self.chunk.popleft()
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

    ax.clear()
    ax.set_yscale('asinh', base=2)
    ax.set_ylim((-scaling, scaling))
    # ax.scatter(ts, ys)
    ax.fill_between(ts, ymins, ys, alpha=1, linewidth=dt)


with open(sys.stdin.fileno(), "rb", closefd=False) as f:
    # needs initial value, s.t. animate doesn't whine on extend
    ys = [0]
    ts = [0]
    ani = animation.FuncAnimation(fig, animate, fargs=(ts, ys), frames=partial(generateData, f),
                                  save_count=displaysize, interval=8)
    plt.show()
