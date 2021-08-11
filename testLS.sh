#!/bin/bash

# Author: Federico Finocchio -- ID: 516818
# email: f.finocchio@studenti.unipi.it
# AA: 2020/2021

# Script file to test the execution of the QR factorization and LS solver.
# Refers to the implementation of the LS class in src/LS/LS.py which is used to
# solve the Least Square problem of the form  min ||Ax - b||.

# NOTE: all the results for the tests performed using this script file will be
#		printed to a specific file with a standard name relative to the performed
#		test. 
# To perform the test just run:
# ./testLS.sh 't' ['m' 'n' 'stepm' 'lastm']

# Where:
#	· t: test type to perform:
#		- 'RANDOM': to test the execution of the two algorithm over a randomly
#					generated set of data with dimensions as specified by the
#					parameters 'm' and 'n', that represents, respectively,
#					the number of rows and columns of the matrix A ;
#
#		- 'SCALING': to test the performance of the two algorithms over increasing
#					 dimension 'm' with a fixed 'n'. Starting from 'm' to 'lastm'
#					 with a 'stepm' increment, test each of the randomly generated
#					 datasets up to A in R^[lastm x n] ;
#
#		- 'CUP':  to test the two algorithms over the dataset provided for the ML
#				  cup, as specified in the report file.
#
#	· m: number of rows for the 'RANDOM' test and starting number of rows for
#		 'SCALING' test. Must be greater than 0;
#
#	· n: number of columns for the matrix A;
#
#	· stepm: increment for the m dimension, to be used only for 'SCALING' test;
#
#	· lastm: last value for the m dimension, to be used only for 'SCALING' test;


t="$1"		# test type
m="$2"		# starting value for m
n="$3"		# starting value for n

REPEAT=10

case $t in
	'RANDOM')
		# for each value of m from 'm' to 'lastm' with step 'stepm'
		python -m src.LS.testLS $t $m $n >> "test_random${m}_${n}.txt"
	;;
	'SCALING')
		stepm="$4"	# step for m
		lastm="$5"	# last value for m
		python -m src.LS.testLS $t $m $lastm $n $stepm $REPEAT >> "test_scaling${lastm}_${n}.txt"
	;;
	'CUP')
		python -m src.LS.testLS $t 'data/ML-CUP20-TR.csv' >> "test_cup.txt"
esac