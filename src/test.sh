#!/bin/bash

m="$1"
n="$2"
stepm="$3"
stepn="$4"
lastm="$5"
lastn="$6"

# for i in $(seq $m $step $last); do for j in $(seq 500 500 $i); do python test.py $i $j >> test.txt; done; done;
for i in $(seq $m $stepm $lastm)
do
	for j in $(seq $n $stepn $lastn)
	do
		python test.py $i $j >> test_script.txt
	done
done
