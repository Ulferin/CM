#!/bin/bash

m="$1"		# starting value for m
n="$2"		# starting value for n
stepm="$3"	# step for m
stepn="$4"	# step for n
lastm="$5"	# last value for m
lastn="$6"	# last value for n

# for each value of m from 'm' to 'lastm' with step 'stepm'
for i in $(seq $m $stepm $lastm)
do
	# for each value of n from 'n' to 'lastn' with step 'lastn'
	for j in $(seq $n $stepn $lastn)
	do
		python LS/testLS.py 'random' $i $j >> "test${lastm}_${lastn}.txt"
	done
done
