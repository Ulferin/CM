#!/bin/bash

# Author: Federico Finocchio
# Author: Luca Santarella
# email: f.finocchio@studenti.unipi.it
# email: l.santarella@studenti.unipi.it
# Course: Computational Mathematics for Learning and Data Analysis
# AY: 2021/2022

# Script file to test the execution of the QR factorization and LS solver.
# Refers to the implementation of the LS class in src/LS/LS.py which is used to
# solve the Least Square problem of the form  min ||Ax - b||.
# In order to generate more accurate time statistics, all the tests performed via
# this script file will be performed for a total of 5 times and the average of
# all the results will be written to the result file in tests/LS/. Moreover,
# plots will be generated for the executed tests and saved in tests/LS/plots.

# NOTE: all the results for the tests performed using this script file will be
#       saved on a specific file with a standard name relative to the performed
#       test and the used configuration.
#       Defaults value for m and n parameters are set to 
# To perform the test just run:
# ./testLS.sh 't' ['m' 'n' 'stepm' 'lastm']

# Where:
#    · t: test type to perform:
#        - 'RANDOM': to test the execution of the two algorithm over a randomly
#                    generated set of data with dimensions as specified by the
#                    parameters 'm' and 'n', that represents, respectively,
#                    the number of rows and columns of the matrix A ;
#                    Generates a file named test_random_<m>_<n>.txt
#
#        - 'SCALING': to test the performance of the two algorithms over increasing
#                     dimension 'm' with a fixed 'n'. Starting from 'm' to 'lastm'
#                     with a 'stepm' increment, test each of the randomly generated
#                     datasets up to A in R^[lastm x n] ;
#                     Generates a file named test_scaling<lastm>_<n>.txt
#
#        - 'CUP':  to test the two algorithms over the dataset provided for the ML
#                  cup, as specified in the report file.
#                  Generates a file named test_cup.txt
#
#    · m: number of rows for the 'RANDOM' test and starting number of rows for
#         'SCALING' test. Must be greater than 0;
#
#    · n: number of columns for the matrix A;
#
#    · stepm: increment for the m dimension, to be used only for 'SCALING' test;
#
#    · lastm: last value for the m dimension, to be used only for 'SCALING' test;


# Example Scaling test
# $ ./testLS.sh 'SCALING' 1000 100 1000 10000
#
# Will perform the 'SCALING' test starting from m=1000 and n=100 up to
# m=10000 and n=100 incrementing the m dimension by 1000 at each new test.
# It will write the recorded statistics to a file named test_scaling10000_100.txt
# including the last m dimension tested and the n dimension in the name, making
# it easier to gather statistics and scrape results from statistics file.


t="$1"        # test type
m="$2"        # starting value for m
n="$3"        # starting value for n

REPEAT=5      # number of times to repeat each test

case $t in
    'RANDOM')
        # for each value of m from 'm' to 'lastm' with step 'stepm'
        python -m src.LS.testLS $t $m $n >> "tests/LS/test_random${m}_${n}.txt"
    ;;
    'SCALING')
        stepm="$4"    # step for m
        lastm="$5"    # last value for m
        python -m src.LS.testLS $t $m $lastm $n $stepm $REPEAT >> "tests/LS/test_scaling${lastm}_${n}.txt"
    ;;
    'CUP')
        python -m src.LS.testLS $t 'data/ML-CUP20-TR.csv' >> "tests/LS/test_cup.txt"
esac
