#!/bin/bash

#python main.py --scenario_name Disjoint --num_tasks 2 --name_algo baseline  --dataset CIFAR10 --pretrained_on CIFAR100 --OutLayer Linear --seed 1
#python main.py --scenario_name Disjoint --num_tasks 2 --name_algo baseline  --dataset CIFAR10 --pretrained_on CIFAR100 --OutLayer CosLayer --seed 1
#python main.py --scenario_name Disjoint --num_tasks 2 --name_algo baseline  --dataset CIFAR10 --pretrained_on CIFAR100 --OutLayer SLDA --seed 1
#python main.py --scenario_name Disjoint --num_tasks 2 --name_algo baseline  --dataset CIFAR10 --pretrained_on CIFAR100 --OutLayer Linear_no_bias --seed 1
#python main.py --scenario_name Disjoint --num_tasks 2 --name_algo baseline  --dataset CIFAR10 --pretrained_on CIFAR100 --OutLayer MeanLayer --seed 1
#python main.py --scenario_name Disjoint --num_tasks 2 --name_algo baseline  --dataset CIFAR10 --pretrained_on CIFAR100 --OutLayer MIMO_Linear --seed 1
#python main.py --scenario_name Disjoint --num_tasks 2 --name_algo baseline  --dataset CIFAR10 --pretrained_on CIFAR100 --OutLayer MIMO_CosLayer --seed 1
#python main.py --scenario_name Disjoint --num_tasks 2 --name_algo baseline  --dataset CIFAR10 --pretrained_on CIFAR100 --OutLayer MIMO_Linear_no_bias --seed 1



#python main.py --scenario_name Disjoint --num_tasks 2 --name_algo baseline  --dataset CIFAR10 --pretrained_on CIFAR100 --OutLayer KNN --seed 0

python main.py --scenario_name Disjoint --num_tasks 2 --name_algo baseline  --dataset CIFAR10 --pretrained_on CIFAR100 --OutLayer Linear --seed 0 --masked
python main.py --scenario_name Disjoint --num_tasks 2 --name_algo baseline  --dataset CIFAR10 --pretrained_on CIFAR100 --OutLayer CosLayer --seed 0 --masked
python main.py --scenario_name Disjoint --num_tasks 2 --name_algo baseline  --dataset CIFAR10 --pretrained_on CIFAR100 --OutLayer Linear_no_bias --seed 0 --masked
python main.py --scenario_name Disjoint --num_tasks 2 --name_algo baseline  --dataset CIFAR10 --pretrained_on CIFAR100 --OutLayer MIMO_Linear --seed 0 --masked
python main.py --scenario_name Disjoint --num_tasks 2 --name_algo baseline  --dataset CIFAR10 --pretrained_on CIFAR100 --OutLayer MIMO_CosLayer --seed 0 --masked
python main.py --scenario_name Disjoint --num_tasks 2 --name_algo baseline  --dataset CIFAR10 --pretrained_on CIFAR100 --OutLayer MIMO_Linear_no_bias --seed 0 --masked

python main.py --scenario_name Disjoint --num_tasks 2 --name_algo baseline  --dataset CIFAR10 --pretrained_on CIFAR100 --OutLayer Linear --seed 1 --masked
python main.py --scenario_name Disjoint --num_tasks 2 --name_algo baseline  --dataset CIFAR10 --pretrained_on CIFAR100 --OutLayer CosLayer --seed 1 --masked
python main.py --scenario_name Disjoint --num_tasks 2 --name_algo baseline  --dataset CIFAR10 --pretrained_on CIFAR100 --OutLayer Linear_no_bias --seed 1 --masked
python main.py --scenario_name Disjoint --num_tasks 2 --name_algo baseline  --dataset CIFAR10 --pretrained_on CIFAR100 --OutLayer MIMO_Linear --seed 1 --masked
python main.py --scenario_name Disjoint --num_tasks 2 --name_algo baseline  --dataset CIFAR10 --pretrained_on CIFAR100 --OutLayer MIMO_CosLayer --seed 1 --masked
python main.py --scenario_name Disjoint --num_tasks 2 --name_algo baseline  --dataset CIFAR10 --pretrained_on CIFAR100 --OutLayer MIMO_Linear_no_bias --seed 1 --masked