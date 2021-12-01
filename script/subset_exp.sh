#!/bin/bash


list_heads_01="CosLayer WeightNorm OriginalWeightNorm"
list_heads_001=" Linear Linear_no_bias"
list_heads_WO_lr="SLDA MeanLayer MedianLayer KNN"

for head in $list_heads_01 ;do
sbatch script/run_on_cluster.sh --lr 0.1 --OutLayer $head
sbatch script/run_on_cluster.sh --lr 0.1 --OutLayer $head --masked_out single
done #head

for head in $list_heads_001 ;do
sbatch script/run_on_cluster.sh --lr 0.01 --OutLayer $head
sbatch script/run_on_cluster.sh --lr 0.01 --OutLayer $head --masked_out single
done #head

for head_WO_lr in $list_heads_WO_lr ;do
sbatch script/run_on_cluster.sh --OutLayer $head_WO_lr
done #head_WO_lr

