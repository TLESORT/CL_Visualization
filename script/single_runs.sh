#!/bin/bash


command="python main.py  --dev --scenario_name Disjoint --num_tasks 1 --name_algo baseline  --dataset Core50 --pretrained_on ImageNet --fast --lr 0.1 --seed 1 --OutLayer Linear --masked_out single --subset 1000 --architecture resnet"
sbatch script/single_on_cluster.sh $command


#command="python main.py  --scenario_name Disjoint --num_tasks 1 --name_algo baseline  --dataset Core50 --pretrained_on ImageNet --fast --lr 0.1 --seed 0 --OutLayer WeightNorm --masked_out single --subset 1000 --architecture resnet "
#sbatch script/single_on_cluster.sh $command


#command="python main.py  --scenario_name Disjoint --num_tasks 1 --name_algo baseline  --dataset Core50 --pretrained_on ImageNet --fast --lr 0.1 --seed 1 --OutLayer WeightNorm --masked_out single --architecture resnet"
#sbatch script/single_on_cluster.sh $command


#command="python main.py  --scenario_name Disjoint --num_tasks 1 --name_algo baseline  --dataset Core10Lifelong --pretrained_on ImageNet --fast --lr 0.1 --seed 1 --OutLayer Linear --masked_out single --subset 1000 --architecture resnet"
#sbatch script/single_on_cluster.sh $command


#command="python main.py  --scenario_name Disjoint --num_tasks 1 --name_algo baseline  --dataset Core10Lifelong --pretrained_on ImageNet --fast --lr 0.1 --seed 0 --OutLayer WeightNorm --masked_out single --subset 1000 --architecture resnet"
#sbatch script/single_on_cluster.sh $command


#command="python main.py  --scenario_name Disjoint --num_tasks 1 --name_algo baseline  --dataset Core10Lifelong --pretrained_on ImageNet --fast --lr 0.1 --seed 0 --OutLayer WeightNorm --subset 1000 --architecture resnet"
#sbatch script/single_on_cluster.sh $command
