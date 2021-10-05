#!/bin/bash


list_subsets="100 200 500 1000"
seeds="0 1 2 3 4 5 6 7"
list_heads_01="CosLayer WeightNorm OriginalWeightNorm"
list_heads_001=" Linear Linear_no_bias"
list_heads_WO_lr="SLDA MeanLayer MedianLayer KNN"

for seed in $seeds ;do

for subset in $list_subsets ;do
for head in $list_heads_01 ;do
sbatch script/run_on_cluster.sh --lr 0.1 --subset $subset --seed $seed --OutLayer $head
sbatch script/run_on_cluster.sh --lr 0.1 --subset $subset --seed $seed --OutLayer $head --masked_out single
done #head

for head in $list_heads_001 ;do
sbatch script/run_on_cluster.sh --lr 0.01 --subset $subset --seed $seed --OutLayer $head
sbatch script/run_on_cluster.sh --lr 0.01 --subset $subset --seed $seed --OutLayer $head --masked_out single
done #head

for head_WO_lr in $list_heads_WO_lr ;do
sbatch script/run_on_cluster.sh --subset $subset --seed $seed --OutLayer $head_WO_lr
done #head_WO_lr

done #subset
done #seed
