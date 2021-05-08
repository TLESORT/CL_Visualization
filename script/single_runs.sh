#!/bin/bash

command="python main.py --scenario_name Domain --num_tasks 8 --name_algo baseline  --dataset Core10Lifelong --pretrained_on ImageNet --fast --seed 1664 --OutLayer Linear --architecture resnet"
sbatch script/single_on_cluster.sh $command
