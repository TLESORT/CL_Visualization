#!/bin/bash

python main.py --scenario_name Disjoint --num_tasks 5 --name_algo baseline  --dataset CIFAR10 --pretrained_on CIFAR10 --fast $@
python main.py --scenario_name Disjoint --num_tasks 5 --name_algo baseline  --dataset CIFAR10 --pretrained_on CIFAR100 --fast $@
python main.py --scenario_name Disjoint --num_tasks 10 --name_algo baseline  --dataset Core50 --pretrained_on ImageNet --architecture resnet --fast $@
python main.py --scenario_name Domaingit --num_tasks 8 --name_algo baseline  --dataset Core10Lifelong --pretrained_on ImageNet --architecture resnet  --fast $@

python main.py --scenario_name Disjoint --num_tasks 1 --name_algo baseline  --dataset CIFAR10 --pretrained_on CIFAR10 --fast $@
python main.py --scenario_name Disjoint --num_tasks 1 --name_algo baseline  --dataset CIFAR10 --pretrained_on CIFAR10 --fast $@
python main.py --scenario_name Disjoint --num_tasks 1 --name_algo baseline  --dataset Core50 --pretrained_on ImageNet --architecture resnet --fast $@