#!/bin/bash

m="$1"		# starting value for m
n="$2"		# starting value for n
stepm="$3"	# step for m
lastm="$4"	# last value for m

# for each value of m from 'm' to 'lastm' with step 'stepm'
for i in $(seq $m $stepm $lastm)
do
	python -m src.LS.testLS 'RANDOM' $i $n >> "test${lastm}_${n}.txt"
done
