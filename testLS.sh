#!/bin/bash

t="$1"		# test type
m="$2"		# starting value for m
n="$3"		# starting value for n
stepm="$4"	# step for m
lastm="$5"	# last value for m

REPEAT=10

# TODO: cambiare tipo di test in base a valore di t:
#		creare test per scaling, test per random e test per CUP
case $t in
	'RANDOM')
		# for each value of m from 'm' to 'lastm' with step 'stepm'
		python -m src.LS.testLS 'RANDOM' $m $n >> "test_random${m}_${n}.txt"
	;;
	'SCALING')
		python -m src.LS.testLS 'SCALING' $m $lastm $n $stepm $REPEAT >> "test_scaling${lastm}_${n}.txt"
	;;
esac