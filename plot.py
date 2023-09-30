import sys
import struct
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.scale as scale
from functools import partial
from math import e
#from collections import deque

plt.style.use('dark_background')
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
displaysize = 32768
bufsize = displaysize << 3
dt = displaysize >> 3
scaling=pow(2,-e)

class Chunker:
    def __init__(self, file):
        self.file = file;
        self.fs = bufsize*'f';
        self.ymins = dt*0;

    def __iter__(self):
        self.chunk = list(struct.unpack_from(self.fs, self.file.read(bufsize << 2)));
        return self;

    def __next__(self):
        if (bool(self.chunk)):
            result = self.chunk[0:dt];
            del self.chunk[0:dt];
            return self.ymins, result;
            #return self.chunk.popleft();
        raise StopIteration();


def generateData(file):

    chunker = Chunker(file);
    return iter(chunker);

def animate(i, xs, ys):

    #if (bool(xs)):
    #    xs.append(xs[-1] + 1);
    #else:
    #    xs.append(1);
    #ys.append(i);
    (ymins, y) = i;
    xs.extend(range(xs[-1],xs[-1]+len(y)));
    ys.extend(y);
    xs = xs[-displaysize:];
    ys = ys[-displaysize:];

    ax.clear();
    ax.set_yscale('asinh', base=2);
    ax.set_ylim((-scaling,scaling));
    #ax.scatter(xs, ys);
    ax.fill_between(xs, ymins, ys, alpha=1, linewidth=dt)

with open(sys.stdin.fileno(), "rb", closefd=False) as f:

    # needs initial value, s.t. animate doesn't whine on extend
    ys = [0];
    xs = [0];
    ani = animation.FuncAnimation(fig, animate, fargs=(xs, ys), frames=partial(generateData, f), save_count=displaysize, interval=8);
    plt.show();
