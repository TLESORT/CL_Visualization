#!/bin/bash


python main.py --scenario_name Disjoint --num_tasks 1 --name_algo baseline  --dataset CIFAR10 --pretrained_on CIFAR10 --fast $@
python main.py --scenario_name Disjoint --num_tasks 1 --name_algo baseline  --dataset CIFAR10 --pretrained_on CIFAR100 --fast $@
python main.py --scenario_name Disjoint --num_tasks 1 --name_algo baseline  --dataset CIFAR100 --pretrained_on CIFAR10 --fast $@
python main.py --scenario_name Disjoint --num_tasks 1 --name_algo baseline  --dataset CIFAR100 --pretrained_on CIFAR100 --fast $@

python main.py --scenario_name Disjoint --num_tasks 1 --name_algo baseline  --dataset Core50 --pretrained_on ImageNet --architecture resnet --fast $@
python main.py --scenario_name Disjoint --num_tasks 1 --name_algo baseline  --dataset Core50 --pretrained_on ImageNet --architecture vgg --fast $@
python main.py --scenario_name Disjoint --num_tasks 1 --name_algo baseline  --dataset Core50 --pretrained_on ImageNet --architecture googlenet --fast $@

# here Core10Lifelong is disjoint because there is only one task
python main.py --scenario_name Disjoint --num_tasks 1 --name_algo baseline  --dataset Core10Lifelong --pretrained_on ImageNet --architecture resnet  --fast $@
python main.py --scenario_name Disjoint --num_tasks 1 --name_algo baseline  --dataset Core10Lifelong --pretrained_on ImageNet --architecture vgg --fast $@
python main.py --scenario_name Disjoint --num_tasks 1 --name_algo baseline  --dataset Core10Lifelong --pretrained_on ImageNet --architecture googlenet --fast $@

