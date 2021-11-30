#!/bin/bash


list_heads="Linear CosLayer Linear_no_bias WeightNorm OriginalWeightNorm"
list_heads_WO_lr="SLDA MeanLayer MedianLayer KNN"

#list_heads="Linear WeightNorm"
#list_heads_WO_lr="SLDA"

lrs="0.1 0.01 0.001"
lrs='0.1 0.01'


name_algos='ewc_diag rehearsal baseline'
name_algos='baseline'

for name_algo in $name_algos ;do

for lr in $lrs ;do

for head in $list_heads ;do
sbatch script/run_on_cluster_continual.sh --name_algo $name_algo --lr $lr --OutLayer $head
sbatch script/run_on_cluster_continual.sh --name_algo $name_algo --lr $lr --OutLayer $head --masked_out single
sbatch script/run_on_cluster_continual.sh --name_algo $name_algo --lr $lr --OutLayer $head --masked_out group
sbatch script/run_on_cluster_continual.sh --name_algo $name_algo --lr $lr --OutLayer $head --masked_out multi-head
done #head

done #lrs

for head_WO_lr in $list_heads_WO_lr ;do
sbatch script/run_on_cluster_continual.sh --OutLayer $head_WO_lr
done #head_WO_lr

done #name_algo
