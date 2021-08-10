#!/bin/bash

t="$1"
dataset="$2"
optimizer="$3"

case $t in
    0)
        python -m src.NN.testNET $optimizer $dataset >> "test_${dataset}_${optimizer}.txt"
    ;;
    1)
        python -m src.NN.testNET $optimizer $dataset 'grid' >> "test_grid_${dataset}_${optimizer}.txt"
    ;;
esac