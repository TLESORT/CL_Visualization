#!/bin/bash

#seeds="0 1 2 3 4 5 6 7"
seeds="0 1"
lrs="0.1 0.01 0.001"
lrs="1.0"
list_subsets="100 200 500 1000 10000 None"
list_subsets="None"
list_heads="CosLayer" #CosLayer WeightNorm SLDA MeanLayer MedianLayer Linear Linear_no_bias KNN"
list_heads_masked="CosLayer" # Linear Linear_no_bias WeightNorm" # MIMO_Linear_no_bias MIMO_Linear MIMO_CosLayer"

#list_heads="CosLayer KNN SLDA MeanLayer MedianLayer"
 

#list_heads=""
#list_heads_masked="Linear"
#list_heads="MeanLayer" #KNN
#list_heads_masked=""

for seed in $seeds ;do

for subset in $list_subsets ;do
for lr in $lrs ;do
for head in $list_heads ;do
sbatch script/run_on_cluster.sh --lr $lr --seed $seed --OutLayer $head #--subset $subset
done #head

#for head_masked in $list_heads_masked ;do
#sbatch script/run_on_cluster.sh --lr $lr --seed $seed --OutLayer $head_masked --masked_out single #--subset $subset
#sbatch script/run_on_cluster.sh --lr $lr --seed $seed --OutLayer $head_masked --masked_out group #--subset $subset
#done #head_masked
done # lrs
done #subset

done #seed
