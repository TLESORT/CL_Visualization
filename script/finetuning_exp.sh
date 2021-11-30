#!/bin/bash

seeds="0 1"
list_heads="Linear CosLayer Linear_no_bias WeightNorm OriginalWeightNorm"
#list_heads="Linear"
#list_heads_WO_lr="SLDA MeanLayer MedianLayer KNN"

name_algos='ewc_diag rehearsal' #baseline


for name_algo in $name_algos ;do
for seed in $seeds ;do


for head in $list_heads ;do
#sbatch script/run_on_cluster_finetuning.sh  --name_algo $name_algo --seed $seed --OutLayer $head --finetuning
#sbatch script/run_on_cluster_finetuning.sh  --name_algo $name_algo --seed $seed --OutLayer $head --masked_out single --finetuning
#sbatch script/run_on_cluster_finetuning.sh  --name_algo $name_algo --seed $seed --OutLayer $head --masked_out group --finetuning
sbatch script/run_on_cluster_finetuning.sh  --name_algo $name_algo --seed $seed --OutLayer $head --masked_out multi-head --finetuning
done #head

done #seed

done #name_algo
