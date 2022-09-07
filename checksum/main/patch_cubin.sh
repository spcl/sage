#!/bin/bash

set -e

if [ $# -ne 4 ]; then
    echo "Usage: $0 src.cubin gen.bin func_name dst.cubin"
    exit 1
fi

SRCCUBIN="$1"
GENBIN="$2"
FUNC="$3"
DSTCUBIN="$4"

SIZE=$(objdump -h $SRCCUBIN | grep $FUNC | tr -s ' ' | cut -f 4 -d ' ')
OFF=$(objdump -h $SRCCUBIN | grep $FUNC | tr -s ' ' | cut -f 7 -d ' ')
cp $SRCCUBIN $DSTCUBIN
dd conv=notrunc if=$GENBIN of=$DSTCUBIN bs=1 seek=$[0x$OFF] count=$[0x$SIZE]
