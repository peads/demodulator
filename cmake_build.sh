#!/bin/bash
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

CUST_OPTS=""

if [ ! -z "$1" ]; then
    CUST_OPTS=$1
fi

rm -rf build/ ||:
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_MAKE_PROGRAM=`which ninja` -DIS_NATIVE=ON $CUST_OPTS -G Ninja -S . -B build
cmake --build build

if [ ! -z "$2" ]; then
    sudo cmake --install build --prefix $2
fi
