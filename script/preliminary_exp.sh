#!/bin/bash

seeds="0 1"
lrs="0.1 0.01 0.001"
list_heads=" Linear CosLayer Linear_no_bias WeightNorm"
list_heads_WO_lr="SLDA MeanLayer MedianLayer KNN"

for seed in $seeds ;do

for lr in $lrs ;do
for head in $list_heads ;do
sbatch script/run_on_cluster.sh --lr $lr --seed $seed --OutLayer $head
done #head
done # lrs


for head_WO_lr in $list_heads_WO_lr ;do
sbatch script/run_on_cluster.sh --seed $seed --OutLayer $head_WO_lr
done #head_WO_lr

done #seed


