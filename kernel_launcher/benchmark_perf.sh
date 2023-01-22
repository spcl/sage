#!/bin/bash

set -e

for b in 1 2 4 8 16 32 64 128 256 512 1024
do
    echo "Start run protected 0 :: args 3 10 $b 1 1 0"
    ./mlp 3 10 $b 1 1 0
    echo "Start run protected 1 :: args 3 10 $b 1 1 0"
    ./mlp_protected 3 10 $b 1 1 0
done