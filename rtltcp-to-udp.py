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
import queue
import socket
import threading
from enum import Enum
from struct import pack
from struct import error as StructError
from contextlib import closing
import typer
import netifaces as ni


# translated directly from rtl_tcp.c
class RtlTcpCommands(Enum):
    SET_FREQUENCY = 0x01
    SET_SAMPLE_RATE = 0x02
    SET_GAIN_MODE = 0x03
    SET_GAIN = 0x04
    SET_FREQUENCY_CORRECTION = 0x05
    SET_IF_STAGE = 0x06
    SET_TEST_MODE = 0x07
    SET_AGC_MODE = 0x08
    SET_DIRECT_SAMPLING = 0x09
    SET_OFFSET_TUNING = 0x0A
    SET_RTL_CRYSTAL = 0x0B
    SET_TUNER_CRYSTAL = 0x0C
    SET_TUNER_GAIN_BY_INDEX = 0x0D
    SET_BIAS_TEE = 0x0E
    SET_TUNER_BANDWIDTH = 0x40
    UDP_ESTABLISH = 0x41
    UDP_TERMINATE = 0x42
    SET_I2C_TUNER_REGISTER = 0x43
    SET_I2C_TUNER_OVERRIDE = 0x44
    SET_TUNER_BW_IF_CENTER = 0x45
    SET_TUNER_IF_MODE = 0x46
    SET_SIDEBAND = 0x47
    REPORT_I2C_REGS = 0x48
    GPIO_SET_OUTPUT_MODE = 0x49
    GPIO_SET_INPUT_MODE = 0x50
    GPIO_GET_IO_STATUS = 0x51
    GPIO_WRITE_PIN = 0x52
    GPIO_READ_PIN = 0x53
    GPIO_GET_BYTE = 0x54
    IS_TUNER_PLL_LOCKED = 0x55
    SET_FREQ_HI32 = 0x56


class UnrecognizedInputError(Exception):
    def __init__(self, msg: str, e: Exception = None):
        super().__init__(f'{msg}, {e}')


class SelbstmortError(Exception):
    def __init__(self):
        super().__init__("It's time to stop!")


class ControlRtlTcp:
    def __init__(self, connection):
        self.connection = connection
        connection.sendall(pack('>BI', 3, 1))
        connection.sendall(pack('>BI', 8, 0))
        connection.sendall(pack('>BI', 14, 0))
        connection.sendall(pack('>BI', 2, 250000))

    def setFrequency(self, freq):
        self.setParam(RtlTcpCommands.SET_FREQUENCY.value, freq)

    def setParam(self, command, param):
        print(f'{RtlTcpCommands(command)}: {param}')
        try:
            self.connection.sendall(pack('>BI', command, param))
        except StructError:
            try:
                self.connection.sendall(pack('>Bi', command, param))
            except StructError as e:
                raise UnrecognizedInputError(param, e)


def findBroadcastAddr(iface='eth0'):
    return ni.ifaddresses(iface)[ni.AF_INET][0]['broadcast']


# taken from https://stackoverflow.com/a/45690594
def findPort(host='localhost'):
    with closing(socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        # s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        s.bind((host, 0))
        # cs.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 0)
        return s.getsockname()[1]


class OutputServer:
    def __init__(self, rs: socket.socket, port: int, host='localhost', bufSize=8192):
        self.rs = rs
        self.cs = None
        self.host = host
        self.port = port
        self.serverhost = findBroadcastAddr()
        self.serverport = findPort(host)
        self.exitFlag = False
        self.bufSize = bufSize
        self.buffer = queue.Queue(maxsize=bufSize)

    def kill(self):
        self.exitFlag = True

    def isNotDead(self):
        return not self.exitFlag

    def consume(self):
        try:
            while self.isNotDead():
                self.cs.sendto(self.buffer.get(block=True), ('', self.serverport))
        except OSError as e:
            print('Consumer quitting', e)
            self.kill()

    def produce(self):
        try:
            while self.isNotDead():
                self.buffer.put(item=self.rs.recv(self.bufSize >> 3), block=True)
        except OSError as e:
            print('Producer quitting', e)
            self.kill()

    def runServer(self):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)) as self.cs:
            # self.cs.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.cs.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            self.cs.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            self.cs.setblocking(0)
            ct = threading.Thread(target=self.consume)
            pt = threading.Thread(target=self.produce)
            pt.start()
            ct.start()

            ct.join()
            pt.join()

    def startServer(self):
        st = threading.Thread(target=self.runServer, args=())
        st.start()
        return st


def main(host: str, port: str, bufSize: int = 16777216):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        iport = int(port)
        s.connect((host, iport))
        cmdr = ControlRtlTcp(s)
        server = OutputServer(s, iport, host='', bufSize=bufSize)
        st = server.startServer()

        try:
            while server.isNotDead():
                try:
                    print('Available commands are: ')
                    print()
                    [print(f'{e.value}\t{e.name}') for e in RtlTcpCommands]
                    print()
                    print(f'Broadcasting on {server.serverhost}:{server.serverport}')
                    print()
                    inp = input(
                        'Provide a space-delimited, command-value pair (e.g. SET_GAIN 1):\n')
                    if len(inp) > 1:
                        try:
                            (cmd, param) = inp.split()
                            try:
                                numCmd = int(cmd)
                                numCmd = RtlTcpCommands(numCmd).value
                            except ValueError:
                                numCmd = RtlTcpCommands[cmd].value
                            cmdr.setParam(numCmd, int(param))
                        except (ValueError, KeyError) as e:
                            raise UnrecognizedInputError(inp, e)
                    elif inp == 'q' or inp == 'Q':
                        server.kill()
                    else:
                        raise UnrecognizedInputError(inp)
                except UnrecognizedInputError as e:
                    print(f'ERROR: Input invalid: {e}. Please try again')
        except SelbstmortError:
            if st is not None:
                print('Joining server thread')
                st.join(timeout=1)
        finally:
            print('Quitting')
            quit(0)


if __name__ == "__main__":
    typer.run(main)
