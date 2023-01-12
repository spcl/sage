#!/bin/bash

set -e

w=5
r=15
c=100
k=100
s=1

for ((b=2; b < 128; b=b*2))
do
    echo "Start run protected 0 :: args $w $r $b $c $k $s"
    ./mlp $w $r $b $c $k $s
    echo "Start run protected 1 :: args $w $r $b $c $k $s"
    ./mlp_protected $w $r $b $c $k $s
done

c=10
k=10

for ((b=128; b < 1024; b=b+128))
do
    echo "Start run protected 0 :: args $w $r $b $c $k $s"
    ./mlp $w $r $b $c $k $s
    echo "Start run protected 1 :: args $w $r $b $c $k $s"
    ./mlp_protected $w $r $b $c $k $s
done

w=1
r=3

for ((b=1024; b < 4096; b=b+512))
do
    echo "Start run protected 0 :: args $w $r $b $c $k $s"
    ./mlp $w $r $b $c $k $s
    echo "Start run protected 1 :: args $w $r $b $c $k $s"
    ./mlp_protected $w $r $b $c $k $s
done
