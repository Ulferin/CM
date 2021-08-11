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
# ./testNN.sh 'dataset' 'optimizer' 'grid'

# Where:
#	· dataset: dataset to use for the test, it can be 'monk1', 'monk2', 'monk3',
#             'cup'. It will use the associated dataset as specified in the report
#              file;
#
#	· optimizer: optimizer to use for the speficied test, either 'SGD' or 'SGM';
#
#
#	· grid: whether to perform a grid search over the specified dataset/optimizer.
#           Either true or false.


dataset="$1"
optimizer="$2"
grid="$3"

if $grid
then
    python -m src.NN.testNET $optimizer $dataset 'grid' >> "test_grid_${dataset}_${optimizer}.txt"
else
    python -m src.NN.testNET $optimizer $dataset >> "test_${dataset}_${optimizer}.txt"
fi