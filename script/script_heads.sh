#!/bin/bash

list_heads="SLDA KNN MeanLayer Linear CosLayer Linear_no_bias"
list_heads_masked="Linear CosLayer Linear_no_bias MIMO_Linear_no_bias MIMO_Linear MIMO_CosLayer"
seed=$1

# CIFAR10 experiments # baseline
for masked_head in $list_heads_masked ;do
python main.py --scenario_name Disjoint --num_tasks 5 --name_algo baseline  --dataset CIFAR10 --pretrained_on CIFAR10 --OutLayer $masked_head --seed $seed --masked --fast
done #list_heads_masked

for head in $list_heads ;do
python main.py --scenario_name Disjoint --num_tasks 5 --name_algo baseline  --dataset CIFAR10 --pretrained_on CIFAR10 --OutLayer $head --seed $seed --fast
done #list_heads

# CIFAR experiments
for masked_head in $list_heads_masked ;do
python main.py --scenario_name Disjoint --num_tasks 5 --name_algo baseline  --dataset CIFAR10 --pretrained_on CIFAR100 --OutLayer $masked_head --seed $seed --masked --fast
done #list_heads_masked

for head in $list_heads ;do
python main.py --scenario_name Disjoint --num_tasks 5 --name_algo baseline  --dataset CIFAR10 --pretrained_on CIFAR100 --OutLayer $head --seed $seed --fast
done #list_heads

# Core50 experiments
for masked_head in $list_heads_masked ;do
python main.py --scenario_name Disjoint --num_tasks 10 --name_algo baseline  --dataset Core50 --pretrained_on ImageNet --OutLayer $masked_head --seed $seed --masked --fast
done #list_heads_masked

for head in $list_heads ;do
python main.py --scenario_name Disjoint --num_tasks 10 --name_algo baseline  --dataset CIFAR10 --pretrained_on ImageNet --OutLayer $head --seed $seed --fast
done #list_heads