#!/bin/bash

seeds="0 1"
list_heads=" Linear CosLayer Linear_no_bias WeightNorm OriginalWeightNorm"
#list_heads_WO_lr="SLDA MeanLayer MedianLayer KNN"


for seed in $seeds ;do


for head in $list_heads ;do
sbatch script/run_on_cluster_finetuning.sh --seed $seed --OutLayer $head --finetuning
sbatch script/run_on_cluster_finetuning.sh --seed $seed --OutLayer $head --masked_out single --finetuning
sbatch script/run_on_cluster_finetuning.sh --seed $seed --OutLayer $head --masked_out group --finetuning
sbatch script/run_on_cluster_finetuning.sh --seed $seed --OutLayer $head --masked_out MHead --finetuning
done #head

#for head_WO_lr in $list_heads_WO_lr ;do
#sbatch script/run_on_cluster_continual.sh --seed $seed --OutLayer $head_WO_lr
#done #head_WO_lr

done #seed