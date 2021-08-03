#!/bin/bash

cd ../..
seeds="0 1 2 3 4 5 6 7"

# baselines
# multi-head
python main.py --dataset mnist_fellowship --scenario_name Disjoint --num_tasks 3 --name_algo baseline --momentum 0.0 --batch_size 32 --seed $seed --test_label --fast
# single-head
python main.py --dataset mnist_fellowship --scenario_name Disjoint --num_tasks 3 --name_algo baseline --momentum 0.0 --batch_size 32 --seed $seed


# rehearsal
# multi-head
python main.py --dataset mnist_fellowship --scenario_name Disjoint --num_tasks 3 --name_algo rehearsal --momentum 0.0 --batch_size 32 --seed $seed  --test_label --fast
# single-head
python main.py --dataset mnist_fellowship --scenario_name Disjoint --num_tasks 3 --name_algo rehearsal --momentum 0.0 --batch_size 32 --seed $seed

# Ewc
# multi-head
python main.py --dataset mnist_fellowship --scenario_name Disjoint --num_tasks 3 --name_algo ewc_diag --momentum 0.0 --batch_size 32 --seed $seed --test_label --importance 1000.0 --fast
# single-head
python main.py --dataset mnist_fellowship --scenario_name Disjoint --num_tasks 3 --name_algo ewc_diag --momentum 0.0 --batch_size 32 --seed $seed --importance 1000.0


# Ewc-kfac
# multi-head
python main.py --dataset mnist_fellowship --scenario_name Disjoint --num_tasks 3 --name_algo ewc_kfac --momentum 0.0 --batch_size 32 --seed $seed --test_label --importance 10.0 --fast
# single-head
python main.py --dataset mnist_fellowship --scenario_name Disjoint --num_tasks 3 --name_algo ewc_kfac --momentum 0.0 --batch_size 32 --seed $seed --importance 10.0

# OGD
# multi-head
python main.py --dataset mnist_fellowship --scenario_name Disjoint --num_tasks 3 --name_algo ogd --momentum 0.0 --batch_size 32 --seed $seed --test_label --fast
# single-head
python main.py --dataset mnist_fellowship --scenario_name Disjoint --num_tasks 3 --name_algo ogd --momentum 0.0 --batch_size 32 --seed $seed

end
