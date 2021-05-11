#!/bin/bash

#seeds="0 1 2 3 4 5 6 7"
seeds="0 1"
lrs="0.1 0.01 0.001"
list_subsets="100 200 500 1000 10000 None"
list_subsets="None"
list_heads="SLDA MeanLayer MedianLayer Linear CosLayer Linear_no_bias KNN"
list_heads_masked="Linear CosLayer Linear_no_bias MIMO_Linear_no_bias MIMO_Linear MIMO_CosLayer"

list_heads="Linear CosLayer Linear_no_bias MIMO_Linear_no_bias MIMO_Linear MIMO_CosLayer"
 

#list_heads=""
#list_heads_masked="Linear"
#list_heads="MeanLayer" #KNN
#list_heads_masked=""

for seed in $seeds ;do

for subset in $list_subsets ;do
for lr in $lrs ;do
for head in $list_heads ;do
sbatch script/run_on_cluster.sh --lr $lr --subset $subset --seed $seed --OutLayer $head 
#./script/test_run.sh $seed
done #head

for head_masked in $list_heads_masked ;do
sbatch script/run_on_cluster.sh --lr $lr --subset $subset --seed $seed --OutLayer $head_masked --masked_out
done #head_masked
done # lrs
done #subset

done #seed
