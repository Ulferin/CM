#!/bin/bash

dataset="$1"
optimizer="$2"
t="$3"

GRID="GRID"

case $t in
    $GRID)
        python -m src.NN.testNET $optimizer $dataset 'grid' >> "test_grid_${dataset}_${optimizer}.txt"
    ;;
    *)
        python -m src.NN.testNET $optimizer $dataset >> "test_${dataset}_${optimizer}.txt"
    ;;
esac