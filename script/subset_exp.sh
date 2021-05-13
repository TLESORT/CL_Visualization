#!/bin/bash

seeds="0 1"
list_subsets="100 200 500 1000 10000 None"


seeds="0 1"
list_heads=" Linear CosLayer Linear_no_bias WeightNorm"
list_heads_WO_lr="SLDA MeanLayer MedianLayer KNN"

for seed in $seeds ;do

for subset in $list_subsets ;do
for head in $list_heads ;do
sbatch script/run_on_cluster_continual.sh --lr 0.1 --subset $subset --seed $seed --OutLayer $head
sbatch script/run_on_cluster_continual.sh --lr 0.1 --subset $subset --seed $seed --OutLayer $head_masked --masked_out single
sbatch script/run_on_cluster_continual.sh --lr 0.1 --subset $subset --seed $seed --OutLayer $head_masked --masked_out group
done #head


for head_WO_lr in $list_heads_WO_lr ;do
sbatch script/run_on_cluster_continual.sh --subset $subset --seed $seed --OutLayer $head_WO_lr
done #head_WO_lr

done #subset
done #seed
