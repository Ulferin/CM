#!/bin/bash

# Author: Federico Finocchio -- ID: 516818
# email: f.finocchio@studenti.unipi.it
# AA: 2020/2021

# Script file to test the fitting of a Neural Network with the specified optimizer
# and dataset. It refers to the implementation of the Network class located in
# src/NN/Network.py. Both normal execution and gridsearch will use the
# configurations as specified in the file src/NN/testNET.py.

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

# Example
# $ ./testNN.sh 'monk1' 'SGD' false
# Will perform the test over the monk1 dataset with the Neural Network using the
# SGD optimizer without performing a grid search. It will produce a file named
# test_monk1_SGD.txt containing all the recorded statistics for the current
# execution.

dataset="$1"
optimizer="$2"
grid="$3"

if $grid
then
    python -m src.NN.testNET $optimizer $dataset 'grid' >> "test_grid_${dataset}_${optimizer}.txt"
else
    python -m src.NN.testNET $optimizer $dataset >> "test_${dataset}_${optimizer}.txt"
fi