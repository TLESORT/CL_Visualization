#!/bin/bash


command="python main.py --scenario_name Disjoint --num_tasks 5 --name_algo baseline --dataset CIFAR10 --pretrained_on CIFAR100 --OutLayer Linear --seed 1 --fast"
sbatch script/single_on_cluster.sh $command

command="python main.py  --scenario_name Disjoint --num_tasks 1 --name_algo baseline  --dataset CIFAR100 --pretrained_on CIFAR10 --fast --seed 0 --OutLayer CosLayer"
sbatch script/single_on_cluster.sh $command

command="python main.py  --scenario_name Disjoint --num_tasks 1 --name_algo baseline  --dataset CIFAR100 --pretrained_on CIFAR10 --fast --seed 1 --OutLayer Linear --masked_out"
sbatch script/single_on_cluster.sh $command


