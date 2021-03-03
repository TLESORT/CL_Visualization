#!/bin/bash

python main.py --dataset mnist_fellowship --scenario_name Disjoint --num_tasks 3 --name_algo rehearsal
python main.py --dataset mnist_fellowship --scenario_name Domain --num_tasks 3 --name_algo rehearsal

python main.py --dataset mnist_fellowship --scenario_name Disjoint --num_tasks 3 --name_algo ewc_diag
python main.py --dataset mnist_fellowship --scenario_name Domain --num_tasks 3 --name_algo ewc_diag

python main.py --dataset mnist_fellowship --scenario_name Disjoint --num_tasks 3 --name_algo ewc_kfac
python main.py --dataset mnist_fellowship --scenario_name Domain --num_tasks 3 --name_algo ewc_kfac
