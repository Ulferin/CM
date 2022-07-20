#!/bin/bash

# Author: Federico Finocchio -- ID: 516818
# Author: Luca Santarella -- ID:
# email: f.finocchio@studenti.unipi.it
# email: l.santarella@studenti.unipi.it
# Course: Computational Mathematics for Learning and Data Analysis
# AY: 2021/2022

# Script file to test the fitting of a Neural Network with the specified optimizer
# and dataset. It tests the implementation of the Network class located in
# src/NN/Network.py. Both normal execution and gridsearch will use the
# configurations as specified in the file src/NN/testNET.py, which reflects the
# best configuration listed in the final report of the CM course exam.

# NOTE: all the results for the tests performed using this script file will be
#       written to a specific file with a standardized name relative to the
#       performed test. 
# To execute the test just run:
# ./testNN.sh 'dataset' 'optimizer' 'grid'

# Where:
#    · dataset: dataset to use for the test, it can be 'monk1', 'monk2', 'monk3',
#               'cup'. It will use the associated dataset as specified in
#               the report file;
#
#    · optimizer: optimizer to use for the speficied test, accepted values
#                are 'CM', 'NAG' and 'Adam' for a test with 'grid'=false,
#                otherwise 'SGD' or 'Adam' are accepted for 'grid'=true;
#
#    · grid: whether to perform a grid search over the specified dataset/optimizer.
#           Either true or false. If true, the script will perform a grid search
#           over the range of parameters contained in src/NN/testNET.py for the
#           specific configuration.

# Usage Example
# $ ./testNN.sh 'monk1' 'CM' false
# Will perform the test over the monk1 dataset with the Neural Network using the
# SGD optimizer with CM update without performing a grid search.
# It will produce a file named test_monk1_CM.txt containing all the recorded
# statistics for the current execution.

# The grid test will also produce a .csv file containing all the tested models
# and the achieved result with the given configuration.
# The path of all the files related to a grid search is src/NN/scores/ and the
# file name for a specific test is <dataset>_<optimizer>.csv

dataset="$1"    # dataset to use for the test
optimizer="$2"  # optimizer to use for the test
grid="$3"       # whether to perform a grid search over the specified setting

if $grid
then
    python -m src.NN.testNET $optimizer $dataset 'grid' > "test_grid_${dataset}_${optimizer}.txt"
else
    python -m src.NN.testNET $optimizer $dataset >> "test_${dataset}_${optimizer}.txt"
fi