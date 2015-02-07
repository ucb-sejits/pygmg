#!/bin/sh
for length in 5 9 17 33 65

do
    for iterations in 10 20 40
    do
        echo
        echo "length:" $length "iterations" $iterations
        ipython benchmarks.py $length $iterations 50
    done
done