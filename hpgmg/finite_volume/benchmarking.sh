#!/bin/sh
for ndim in 2 3
do
for length in 5 9 17

do
    for iterations in 10 20 40 80
    do
        ipython benchmarks.py $length $iterations $ndim 50
    done
done
done