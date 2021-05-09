#!/bin/bash

list_subsets="100 200 500 1000 10000"

for subset in $list_subsets ;do

python main.py --scenario_name Disjoint --num_tasks 1 --name_algo baseline  --dataset CIFAR10 --pretrained_on CIFAR10 --fast --subset $subset $@
python main.py --scenario_name Disjoint --num_tasks 1 --name_algo baseline  --dataset CIFAR10 --pretrained_on CIFAR100 --fast --subset $subset $@
python main.py --scenario_name Disjoint --num_tasks 1 --name_algo baseline  --dataset CIFAR100 --pretrained_on CIFAR10 --fast --subset $subset $@
python main.py --scenario_name Disjoint --num_tasks 1 --name_algo baseline  --dataset CIFAR100 --pretrained_on CIFAR100 --fast --subset $subset $@
python main.py --scenario_name Disjoint --num_tasks 1 --name_algo baseline  --dataset Core50 --pretrained_on ImageNet --architecture resnet --subset $subset --fast $@
python main.py --scenario_name Domaingit --num_tasks 1 --name_algo baseline  --dataset Core10Lifelong --pretrained_on ImageNet --architecture resnet --subset $subset  --fast $@

done #list_subsets
