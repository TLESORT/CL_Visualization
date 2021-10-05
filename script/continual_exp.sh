#!/bin/bash

seeds="0 1"
list_heads="Linear CosLayer Linear_no_bias WeightNorm OriginalWeightNorm"
list_heads_WO_lr="SLDA MeanLayer MedianLayer KNN"

lrs="0.1 0.01 0.001"

for seed in $seeds ;do

for lr in $lrs ;do

for head in $list_heads ;do
sbatch script/run_on_cluster_continual.sh --lr $lr --seed $seed --OutLayer $head
sbatch script/run_on_cluster_continual.sh --lr $lr --seed $seed --OutLayer $head --masked_out single
sbatch script/run_on_cluster_continual.sh --lr $lr --seed $seed --OutLayer $head --masked_out group
sbatch script/run_on_cluster_continual.sh --lr $lr --seed $seed --OutLayer $head --masked_out MHead
done #head

done #lrs


for head_WO_lr in $list_heads_WO_lr ;do
sbatch script/run_on_cluster_continual.sh --seed $seed --OutLayer $head_WO_lr
done #head_WO_lr

done #seed
