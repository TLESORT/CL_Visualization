#!/bin/bash

lrs="0.1 0.01 0.001"

for lr in $lrs ;do

python main.py --lr $lr --scenario_name Disjoint --num_tasks 5 --dataset CIFAR10 --pretrained_on CIFAR100 --fast $@
####python main.py --lr $lr --scenario_name Disjoint --num_tasks 5 --dataset CIFAR100 --pretrained_on CIFAR10 --fast $@
python main.py --lr $lr --scenario_name Disjoint --num_tasks 10 --dataset Core50 --pretrained_on ImageNet --architecture resnet --fast $@
python main.py --lr $lr --scenario_name Domain --num_tasks 8 --dataset Core10Lifelong --pretrained_on ImageNet --architecture resnet  --fast $@
####python main.py --lr $lr --scenario_name Disjoint --num_tasks 10 --dataset Core50 --pretrained_on ImageNet --architecture vgg --fast $@
#####python main.py --lr $lr --scenario_name Domain --num_tasks 8 --dataset Core10Lifelong --pretrained_on ImageNet --architecture vgg  --fast $@

python main.py --lr $lr --scenario_name Domain --num_tasks 50 --dataset Core10Mix --pretrained_on ImageNet --architecture resnet --fast $@

done #lrs
