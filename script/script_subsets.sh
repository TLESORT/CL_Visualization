#!/bin/bash

list_subsets="100 200 500 1000 10000"
seed=1

for subset in $list_subsets ;do
python main.py --scenario_name Disjoint --num_tasks 1 --name_algo baseline  --dataset CIFAR10 --pretrained_on CIFAR100 --OutLayer Linear --seed $seed --subset $subset
python main.py --scenario_name Disjoint --num_tasks 1 --name_algo baseline  --dataset CIFAR10 --pretrained_on CIFAR100 --OutLayer Linear --seed $seed --masked --subset $subset
python main.py --scenario_name Disjoint --num_tasks 1 --name_algo baseline  --dataset CIFAR10 --pretrained_on CIFAR100 --OutLayer MIMO_Linear --seed $seed --masked --subset $subset

python main.py --scenario_name Disjoint --num_tasks 1 --name_algo baseline  --dataset CIFAR10 --pretrained_on CIFAR100 --OutLayer CosLayer --seed $seed --subset $subset
python main.py --scenario_name Disjoint --num_tasks 1 --name_algo baseline  --dataset CIFAR10 --pretrained_on CIFAR100 --OutLayer CosLayer --seed $seed --masked --subset $subset
python main.py --scenario_name Disjoint --num_tasks 1 --name_algo baseline  --dataset CIFAR10 --pretrained_on CIFAR100 --OutLayer MIMO_CosLayer --seed $seed --masked --subset $subset

python main.py --scenario_name Disjoint --num_tasks 1 --name_algo baseline  --dataset CIFAR10 --pretrained_on CIFAR100 --OutLayer Linear_no_bias --seed $seed --subset $subset
python main.py --scenario_name Disjoint --num_tasks 1 --name_algo baseline  --dataset CIFAR10 --pretrained_on CIFAR100 --OutLayer Linear_no_bias --seed $seed --masked --subset $subset
python main.py --scenario_name Disjoint --num_tasks 1 --name_algo baseline  --dataset CIFAR10 --pretrained_on CIFAR100 --OutLayer MIMO_Linear_no_bias --seed $seed --masked --subset $subset

python main.py --scenario_name Disjoint --num_tasks 1 --name_algo baseline  --dataset CIFAR10 --pretrained_on CIFAR100 --OutLayer SLDA --seed $seed --subset $subset
python main.py --scenario_name Disjoint --num_tasks 1 --name_algo baseline  --dataset CIFAR10 --pretrained_on CIFAR100 --OutLayer KNN --seed $seed --subset $subset
python main.py --scenario_name Disjoint --num_tasks 1 --name_algo baseline  --dataset CIFAR10 --pretrained_on CIFAR100 --OutLayer MeanLayer --seed $seed --subset $subset
done #subset