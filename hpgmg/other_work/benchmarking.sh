#!/bin/sh
for ndim in 2 3
do
for length in 5 9 17

do
    for iterations in 40 60 80
    do
        ipython benchmarks.py $length $iterations $ndim 3
    done
done
done