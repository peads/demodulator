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
import queue
import socket
import threading
from enum import Enum
from struct import pack
from struct import error as StructError
from contextlib import closing
import typer
from collections import deque


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
        except StructError as e:
            try:
                self.connection.sendall(pack('>Bi', command, param))
            except StructError as e:
                raise UnrecognizedInputError(param, e)


# taken from https://stackoverflow.com/a/45690594
def findPort(host='localhost'):
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind((host, 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


class GoodNightMoonError(Exception):
    def __init__(self):
        super().__init__('You died')


class OutputServer:
    def __init__(self, rs: socket.socket, port: int, host='localhost', serverhost='localhost'):
        self.ss = None
        self.rs = rs
        self.host = host
        self.port = port
        self.serverport = findPort(host)
        self.serverhost = serverhost
        self.exitFlag = False
        self.buffer = queue.Queue(maxsize=1)
        self.clients = deque()

    def kill(self):
        if self.exitFlag:
            raise Exception('Already dead')
        else:
            self.exitFlag = True
            self.ss.shutdown(socket.SHUT_RDWR)
            self.rs.shutdown(socket.SHUT_RDWR)
            while len(self.clients):
                self.clients.pop().shutdown(socket.SHUT_RDWR)
            self.clients.clear()
            self.buffer = None
            self.clients = None

    def isNotDead(self):
        if not self.exitFlag:
            return True
        raise GoodNightMoonError()

    def consume(self):
        processingList = []
        cs = None
        try:
            while self.isNotDead():
                data = self.buffer.get()

                while self.isNotDead() and len(self.clients):
                    cs = self.clients.pop()
                    try:
                        cs.sendall(data)
                        processingList.append(cs)
                    except OSError as ex:
                        cs.close()
                        print(f'Client disconnected {ex}')
                self.buffer.task_done()
                self.clients.extend(processingList)
                processingList.clear()
        except GoodNightMoonError:
            pass
        except (OSError, queue.Empty, AttributeError, TypeError) as ex:
            print(f'Consumer caught {ex}')
        finally:
            print('Consumer quitting')
            return

    def produce(self):
        try:
            while self.isNotDead():
                self.buffer.join()
                self.buffer.put(item=self.rs.recv(8192))
        except GoodNightMoonError:
            pass
        except (OSError, queue.Full, AttributeError, TypeError) as ex:
            print(f'Producer caught {ex}')
        finally:
            print(f'Producer quitting')
            return

    def runServer(self):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as self.ss:
            self.ss.bind((self.host, self.serverport))
            self.ss.listen(1)
            pt = threading.Thread(target=self.produce, args=())
            pt.start()
            ct = threading.Thread(target=self.consume)
            ct.start()
            try:
                while self.isNotDead():
                    (cs, address) = self.ss.accept()
                    cs.setblocking(False)
                    print(f'Connection request from: {address}')
                    self.clients.appendleft(cs)
            except GoodNightMoonError:
                pass
            finally:
                print('Listener quitting')
                return

    def startServer(self):
        st = threading.Thread(target=self.runServer, args=())
        st.start()
        return st


def main(host: str, port: str):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        iport = int(port)
        s.connect((host, iport))
        cmdr = ControlRtlTcp(s)
        server = OutputServer(s, iport, host='0.0.0.0')
        st = server.startServer()
        # cmdr.setFrequency(int(freq))
        while server.isNotDead():
            try:
                print('Available commands are: ')
                print()
                [print(f'{e.value}\t{e.name}') for e in RtlTcpCommands]
                print()
                print(f'Accepting connections on port {server.serverport}')
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
                    except (ValueError, KeyError) as ex:
                        raise UnrecognizedInputError(inp, ex)
                elif inp == 'q' or inp == 'Q':
                    server.kill()
                else:
                    raise UnrecognizedInputError(inp)
            except UnrecognizedInputError as ex:
                print(f'ERROR: Input invalid: {ex}. Please try again')
        print('Commander quitting')


if __name__ == "__main__":
    try:
        typer.run(main)
    except Exception as e:
        print(f'This was your fate: {e}')
