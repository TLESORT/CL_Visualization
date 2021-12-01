#!/bin/bash

seeds="0 1 2 3 4 5 6 7"
#seeds='8 9 10 11 12 13 14 15'

# / ! \ do not forget to uncomment dataset transferts in run_on_cluster_continual.sh

for seed in $seeds ;do

#python main.py --seed $seed --scenario_name Disjoint --num_tasks 5 --dataset CIFAR10 --pretrained_on CIFAR100 --fast $@

##python main.py --seed $seed --scenario_name Disjoint --num_tasks 5 --name_algo baseline  --dataset CIFAR100 --pretrained_on CIFAR10 --fast $@
#python main.py --seed $seed --scenario_name Disjoint --num_tasks 10 --dataset Core50 --pretrained_on ImageNet --architecture resnet --fast $@

#python main.py --seed $seed --scenario_name Domain --num_tasks 8 --dataset Core10Lifelong --pretrained_on ImageNet --architecture resnet  --fast $@
##python main.py --seed $seed --scenario_name Disjoint --num_tasks 10 --name_algo baseline  --dataset Core50 --pretrained_on ImageNet --architecture vgg --fast $@
##python main.py --seed $seed --scenario_name Domain --num_tasks 8 --name_algo baseline  --dataset Core10Lifelong --pretrained_on ImageNet --architecture vgg  --fast $@

#python main.py --seed $seed --scenario_name Domain --num_tasks 50 --dataset Core10Mix --pretrained_on ImageNet --architecture resnet --fast $@

python main.py --seed $seed --scenario_name Disjoint --num_tasks 10 --dataset CUB200 --pretrained_on ImageNet --architecture resnet --fast $@

##python main.py --seed $seed --scenario_name Domain --num_tasks 5 --dataset CIFAR100Lifelong --pretrained_on CIFAR100 --fast $@
done #seeds
