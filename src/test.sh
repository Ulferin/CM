#!/bin/bash

m="$1"
n="$2"
step="$3"
last="$4"

# for i in $(seq $m $step $last); do for j in $(seq 500 500 $i); do python test.py $i $j >> test.txt; done; done;
for i in $(seq $m $step $last)
do
	for j in $(seq $n $step $i)
	do
		python test.py $i $j >> test_script.txt
	done
done
