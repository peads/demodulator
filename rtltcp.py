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
import typer


# translated directly from rtl_tcp.c
class RtlTcpCommands(Enum):
    SET_FREQUENCY = 0x01 # TODO split this with top half using 0x56?
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


class OutputServer:
    def __init__(self, rs: socket.socket, port: int, host='localhost', bufSize=8192):
        self.rs = rs
        self.host = host
        self.port = port
        self.exitFlag = False
        self.buffer = queue.Queue(maxsize=bufSize)

    def consume(self, cs):
        while not self.exitFlag:
            cs.sendall(self.buffer.get(block=True))

    def produce(self):
        while not self.exitFlag:
            self.buffer.put(item=self.rs.recv(8192), block=True)

    def startServer(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as ss:
            ss.bind((self.host, self.port + 1))
            ss.listen(2)
            ct = None
            pt = None
            while not self.exitFlag:
                print(f'Awaiting connection on port {self.port + 1}')
                (cs, address) = ss.accept()
                print(f'Connected to: {address}')

                ct = threading.Thread(target=self.consume, args=(self, cs))
                pt = threading.Thread(target=self.produce, args=(self,))
                pt.start()
                ct.start()
            ct.join(1)
            pt.join(1)


def main(host: str, port: str):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        iport = int(port)
        s.connect((host, iport))
        cmdr = ControlRtlTcp(s)
        server = OutputServer(s, iport, host='0.0.0.0')
        st = threading.Thread(target=server.startServer, args=(server,))
        st.start()
        # cmdr.setFrequency(int(freq))
        while not server.exitFlag:
            try:
                inp = input('Provide a space-delimited command and value for rtl_tcp:\n')
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
                    server.exitFlag = True
                else:
                    raise UnrecognizedInputError(inp)
            except UnrecognizedInputError as e:
                print(f'ERROR: Input invalid: {e}. Please try again')
        st.join(timeout=1)
        quit(int(st.is_alive()))


if __name__ == "__main__":
    typer.run(main)
