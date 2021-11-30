#!/bin/bash

# ImageNet Experiments
#$SLURM_TMPDIR
#python main.py --scenario_name Disjoint --num_tasks 1 --name_algo baseline  --dataset ImageNet --pretrained_on ImageNet --data_dir $SLURM_TMPDIR --dev

seed=$1

#python main.py --scenario_name Disjoint --num_tasks 10 --name_algo baseline  --dataset Core50 --pretrained_on ImageNet --OutLayer Linear --seed $seed
#python main.py --scenario_name Disjoint --num_tasks 10 --name_algo baseline  --dataset Core50 --pretrained_on ImageNet --OutLayer Linear --seed $seed --masked
#python main.py --scenario_name Disjoint --num_tasks 10 --name_algo baseline  --dataset Core50 --pretrained_on ImageNet --OutLayer MIMO_Linear --seed $seed --masked


#python main.py --scenario_name Disjoint --num_tasks 10 --name_algo baseline  --dataset Core50 --pretrained_on ImageNet --OutLayer Coslayer --seed $seed
#python main.py --scenario_name Disjoint --num_tasks 10 --name_algo baseline  --dataset Core50 --pretrained_on ImageNet --OutLayer Coslayer --seed $seed --masked
#python main.py --scenario_name Disjoint --num_tasks 10 --name_algo baseline  --dataset Core50 --pretrained_on ImageNet --OutLayer MIMO_Coslayer --seed $seed --masked

#python main.py --scenario_name Disjoint --num_tasks 10 --name_algo baseline  --dataset Core50 --pretrained_on ImageNet --OutLayer Linear_no_bias --seed $seed
#python main.py --scenario_name Disjoint --num_tasks 10 --name_algo baseline  --dataset Core50 --pretrained_on ImageNet --OutLayer Linear_no_bias --seed $seed --masked
#python main.py --scenario_name Disjoint --num_tasks 10 --name_algo baseline  --dataset Core50 --pretrained_on ImageNet --OutLayer MIMO_Linear_no_bias --seed $seed --masked

python main.py --scenario_name Disjoint --num_tasks 10 --name_algo baseline  --dataset Core50 --pretrained_on ImageNet --OutLayer MeanLayer --seed $seed
python main.py --scenario_name Disjoint --num_tasks 10 --name_algo baseline  --dataset Core50 --pretrained_on ImageNet --OutLayer KNN --seed $seed
python main.py --scenario_name Disjoint --num_tasks 10 --name_algo baseline  --dataset Core50 --pretrained_on ImageNet --OutLayer SLDA --seed $seed





#for seed in $seeds ;do
#python main.py --dataset mnist_fellowship --scenario_name Disjoint --num_tasks 3 --name_algo baseline --momentum 0.0 --batch_size 32 --seed $seed --test_label --fast
#python main.py --dataset mnist_fellowship --scenario_name Disjoint --num_tasks 3 --name_algo baseline --momentum 0.0 --batch_size 32 --seed $seed --analysis
#python main.py --dataset mnist_fellowship --scenario_name Domain --num_tasks 3 --name_algo baseline --momentum 0.0

#python main.py --dataset mnist_fellowship --scenario_name Disjoint --num_tasks 3 --name_algo rehearsal --momentum 0.0 --batch_size 32 --seed $seed  --test_label --fast

#python main.py --dataset mnist_fellowship --scenario_name Disjoint --num_tasks 3 --name_algo rehearsal --momentum 0.0 --batch_size 32 --seed $seed --analysis

#python main.py --dataset mnist_fellowship --scenario_name Domain --num_tasks 3 --name_algo rehearsal --momentum 0.0

#python main.py --dataset mnist_fellowship --scenario_name Disjoint --num_tasks 3 --name_algo ewc_diag --momentum 0.0 --batch_size 32 --seed $seed --test_label --importance 1000.0 --fast
#python main.py --dataset mnist_fellowship --scenario_name Disjoint --num_tasks 3 --name_algo ewc_diag --momentum 0.0 --batch_size 32 --seed $seed --importance 1000.0 --analysis
#python main.py --dataset mnist_fellowship --scenario_name Domain --num_tasks 3 --name_algo ewc_diag --momentum 0.0

#python main.py --dataset mnist_fellowship --scenario_name Disjoint --num_tasks 3 --name_algo ewc_diag_id --momentum 0.0 --batch_size 32
#python main.py --dataset mnist_fellowship --scenario_name Domain --num_tasks 3 --name_algo ewc_diag_id --momentum 0.0

#python main.py --dataset mnist_fellowship --scenario_name Disjoint --num_tasks 3 --name_algo ewc_kfac --momentum 0.0 --batch_size 32 --seed $seed --test_label --importance 10.0 --fast
#python main.py --dataset mnist_fellowship --scenario_name Disjoint --num_tasks 3 --name_algo ewc_kfac --momentum 0.0 --batch_size 32 --seed $seed --importance 10.0 --analysis
#python main.py --dataset mnist_fellowship --scenario_name Domain --num_tasks 3 --name_algo ewc_kfac --momentum 0.0


#python main.py --dataset mnist_fellowship --scenario_name Disjoint --num_tasks 3 --name_algo ogd --momentum 0.0 --batch_size 32 --seed $seed --test_label --fast
#python main.py --dataset mnist_fellowship --scenario_name Disjoint --num_tasks 3 --name_algo ogd --momentum 0.0 --batch_size 32 --seed $seed --fast
#done # seed
