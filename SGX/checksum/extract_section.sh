#!/bin/bash

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 input_elf output_bin section"
    exit 2
fi

# from here https://stackoverflow.com/a/3925586/8044236
IN_F=$1
OUT_F=$2
SECTION=$3

objdump -h $IN_F | grep $SECTION | awk '{print "dd if='$IN_F' of='$OUT_F' bs=1 count=$[0x" $3 "] skip=$[0x" $6 "]"}' | bash
