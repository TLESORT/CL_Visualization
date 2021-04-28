#!/bin/bash

list_heads="SLDA KNN MeanLayer Linear CosLayer Linear_no_bias"
list_heads_masked="Linear CosLayer Linear_no_bias MIMO_Linear_no_bias MIMO_Linear MIMO_CosLayer"
list_subsets="100 200 500 1000 10000"
seed=$1

for subset in $list_subsets ;do

for masked_head in $list_heads_masked ;do
python main.py --scenario_name Disjoint --num_tasks 1 --name_algo baseline  --dataset CIFAR10 --pretrained_on CIFAR100 --OutLayer $masked_head --seed $seed --masked --subset $subset
done #list_heads_masked

for head in $list_heads ;do
python main.py --scenario_name Disjoint --num_tasks 1 --name_algo baseline  --dataset CIFAR10 --pretrained_on CIFAR100 --OutLayer $head --seed $seed --subset $subset
done #list_heads
done #list_subsets