#!/bin/bash

CUST_OPTS=""
INSTALL_PREFIX="/usr/local"

if [ ! -z "$1" ]; then
    CUST_OPTS=$1
fi

if [ ! -z $2 ]; then
    INSTALL_PREFIX=$2
fi

echo $INSTALL_PREFIX
echo $CUST_OPTS

rm -rf build/ ||:
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release $CUST_OPTS -S . -B build
cmake --build build
cmake --install build --prefix $INSTALL_PREFIX