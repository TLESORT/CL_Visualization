#!/bin/bash

python main.py --dataset mnist_fellowship --scenario_name Disjoint --num_tasks 3 --name_algo baseline --momentum 0.0
#python main.py --dataset mnist_fellowship --scenario_name Domain --num_tasks 3 --name_algo baseline --momentum 0.0

python main.py --dataset mnist_fellowship --scenario_name Disjoint --num_tasks 3 --name_algo rehearsal --momentum 0.0
#python main.py --dataset mnist_fellowship --scenario_name Domain --num_tasks 3 --name_algo rehearsal --momentum 0.0

python main.py --dataset mnist_fellowship --scenario_name Disjoint --num_tasks 3 --name_algo ewc_diag --momentum 0.0
#python main.py --dataset mnist_fellowship --scenario_name Domain --num_tasks 3 --name_algo ewc_diag --momentum 0.0

python main.py --dataset mnist_fellowship --scenario_name Disjoint --num_tasks 3 --name_algo ewc_diag_id --momentum 0.0
#python main.py --dataset mnist_fellowship --scenario_name Domain --num_tasks 3 --name_algo ewc_diag_id --momentum 0.0

python main.py --dataset mnist_fellowship --scenario_name Disjoint --num_tasks 3 --name_algo ewc_kfac --momentum 0.0
#python main.py --dataset mnist_fellowship --scenario_name Domain --num_tasks 3 --name_algo ewc_kfac --momentum 0.0


python main.py --dataset mnist_fellowship --scenario_name Disjoint --num_tasks 3 --name_algo ogd --momentum 0.0