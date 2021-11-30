#!/bin/bash

list_subsets="100 200 500 1000"
seeds="0 1 2 3 4 5 6 7"

for seed in $seeds ;do
#for subset in $list_subsets ;do

##python main.py --subset $subset --seed $seed --scenario_name Disjoint --num_tasks 1 --name_algo baseline  --dataset CIFAR10 --pretrained_on CIFAR10 --fast $@
#python main.py --subset $subset --seed $seed --scenario_name Disjoint --num_tasks 1 --name_algo baseline  --dataset CIFAR10 --pretrained_on CIFAR100 --fast $@
##python main.py --subset $subset --seed $seed --scenario_name Disjoint --num_tasks 1 --name_algo baseline  --dataset CIFAR100 --pretrained_on CIFAR10 --fast $@
##python main.py --subset $subset --seed $seed --scenario_name Disjoint --num_tasks 1 --name_algo baseline  --dataset CIFAR100 --pretrained_on CIFAR100 --fast $@

#python main.py --subset $subset --seed $seed --scenario_name Disjoint --num_tasks 1 --name_algo baseline  --dataset Core50 --pretrained_on ImageNet --architecture resnet --fast $@
##python main.py --subset $subset --seed $seed --scenario_name Disjoint --num_tasks 1 --name_algo baseline  --dataset Core50 --pretrained_on ImageNet --architecture vgg --fast $@
##python main.py --subset $subset --seed $seed --scenario_name Disjoint --num_tasks 1 --name_algo baseline  --dataset Core50 --pretrained_on ImageNet --architecture googlenet --fast $@

## here Core10Lifelong is disjoint because there is only one task
#python main.py --subset $subset --seed $seed --scenario_name Disjoint --num_tasks 1 --name_algo baseline  --dataset Core10Lifelong --pretrained_on ImageNet --architecture resnet --fast $@
##python main.py --subset $subset --seed $seed --scenario_name Disjoint --num_tasks 1 --name_algo baseline  --dataset Core10Lifelong --pretrained_on ImageNet --architecture vgg --fast $@
##python main.py --subset $subset --seed $seed --scenario_name Disjoint --num_tasks 1 --name_algo baseline  --dataset Core10Lifelong --pretrained_on ImageNet --architecture googlenet --fast $@

#python main.py --subset $subset --seed $seed --scenario_name Disjoint --num_tasks 1 --name_algo baseline  --dataset CIFAR100Lifelong --pretrained_on CIFAR100 --architecture resnet --fast $@

#python main.py --subset $subset --seed $seed --scenario_name Disjoint --num_tasks 1 --name_algo baseline  --dataset CUB200 --pretrained_on ImageNet --architecture resnet --fast $@
#python main.py --subset $subset --seed $seed --scenario_name Disjoint --num_tasks 1 --name_algo baseline  --dataset CUB200 --pretrained_on ImageNet --architecture vgg --fast $@
#python main.py --subset $subset --seed $seed --scenario_name Disjoint --num_tasks 1 --name_algo baseline  --dataset CUB200 --pretrained_on ImageNet --architecture googlenet --fast $@

#done #subset

# exp with all data

#python main.py --seed $seed --scenario_name Disjoint --num_tasks 1 --name_algo baseline  --dataset CIFAR10 --pretrained_on CIFAR100 --fast $@

#python main.py --seed $seed --scenario_name Disjoint --num_tasks 1 --name_algo baseline  --dataset Core50 --pretrained_on ImageNet --architecture resnet --fast $@

#python main.py --seed $seed --scenario_name Disjoint --num_tasks 1 --name_algo baseline  --dataset Core10Lifelong --pretrained_on ImageNet --architecture resnet --fast $@

#python main.py --seed $seed --scenario_name Disjoint --num_tasks 1 --name_algo baseline  --dataset CIFAR100Lifelong --pretrained_on CIFAR100 --architecture resnet --fast $@
python main.py --seed $seed --scenario_name Disjoint --num_tasks 1 --name_algo baseline  --dataset CUB200 --pretrained_on ImageNet --architecture resnet --fast $@

done #seeds
