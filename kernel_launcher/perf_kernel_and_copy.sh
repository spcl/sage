#!/bin/bash

set -e

for b in 1 2 4 8 16 32 64 128 256 512 1024
do
    echo "Start run protected 0 :: args 3 10 $b 10 1000 1"
    ./mlp 3 10 $b 10 1000 1
    echo "Start run protected 1 :: args 3 10 $b 10 1000 1"
    ./mlp_protected 3 10 $b 10 1000 1
done