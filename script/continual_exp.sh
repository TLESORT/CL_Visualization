#!/bin/bash

seeds="0 1"
list_heads=" Linear CosLayer Linear_no_bias WeightNorm"
list_heads_WO_lr="SLDA MeanLayer MedianLayer KNN"

for seed in $seeds ;do

for head in $list_heads ;do
sbatch script/run_on_cluster.sh --lr 0.1 --seed $seed --OutLayer $head
sbatch script/run_on_cluster.sh --lr 0.1 --seed $seed --OutLayer $head_masked --masked_out single
sbatch script/run_on_cluster.sh --lr 0.1 --seed $seed --OutLayer $head_masked --masked_out group
done #head


for head_WO_lr in $list_heads_WO_lr ;do
sbatch script/run_on_cluster.sh --seed $seed --OutLayer $head_WO_lr
done #head_WO_lr

done #seed
