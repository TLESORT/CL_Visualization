#!/bin/bash

python main.py  --scenario_name Disjoint --num_tasks 5 --name_algo baseline  --dataset CIFAR10 --pretrained_on CIFAR100 --fast --lr 0.1 --seed 1664 --OutLayer Linear --architecture resnet --finetuning

python main.py  --scenario_name Disjoint --num_tasks 5 --name_algo baseline  --dataset CIFAR10 --pretrained_on CIFAR100 --fast --lr 0.1 --seed 1664 --OutLayer Linear --architecture resnet