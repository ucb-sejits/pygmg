#! /bin/bash

mkdir timings

for i in `seq 5 9`;
do
    ./run_finite_volume $i -b c -v -pn p4 -nv 10 -dre &> timings/snowflake_$i.log
done

grep "solve.*calls" timings/snowflake_*.log | cut -f 4 > data.out

rm -rf timings