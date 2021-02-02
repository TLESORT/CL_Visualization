
import os
import argparse
from model import Model
from trainer import Trainer


# le gradient depend de la couche de sortie (pas toujours)
# mais surtout de la loss function
# c'est l'occasion de trouver des loss / couche de sortie plus adaptées aux continual learning

parser = argparse.ArgumentParser()
parser.add_argument('--name_algo', type=str,
                    choices=['baseline', 'rehearsal', 'ewc_diag'],
                    default='baseline',
                    help='EWC type')
parser.add_argument('--scenario_name', type=str, choices=['Disjoint', 'Rotations'], default="Disjoint",
                    help='continual scenario')
parser.add_argument('--num_tasks', type=int,
                    default='5',
                    help='Task number')
parser.add_argument('--Root_dir', default="./Archives", type=str,
                    help='data directory name')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--importance', default=1.0, type=float, help='Importance of penalty')
parser.add_argument('--epochs', default=10, type=int,
                    help='Epochs for each task')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--num_task', default=3, type=int, help='number of task')
parser.add_argument('--test_label', action='store_true', default=False,
                    help='define if we use task label at test')
parser.add_argument('--expert', action='store_true', default=False,
                    help='define if we use expert model who has access to all data')
parser.add_argument('--dev', action='store_true', default=False, help='dev flag')
parser.add_argument('--dataset', default="mnist_fellowship", type=str,
                    choices=['mnist_fellowship', 'mnist_fellowship_merge'], help='dataset name')
parser.add_argument('--seed', default="1992", type=int,
                    help='seed for number generator')

args = parser.parse_args()

dataset = "MNIST"
root_dir = "./Archives"

model = Model().cuda()

if args.name_algo == "baseline":
    Algo = Trainer(root_dir, dataset, args.scenario_name, model, args.num_tasks, args.dev)
elif args.name_algo == "rehearsal":
    from rehearsal import Rehearsal
    Algo = Rehearsal(root_dir, dataset, args.scenario_name, model, args.num_tasks, args.dev)
elif args.name_algo == "ewc_diag":
    from ewc_diag import EWC_Diag
    Algo = EWC_Diag(root_dir, dataset, args.scenario_name, model, args.num_tasks, args.dev)
else:
    print("wrong name")

Algo.continual_training()
# Algo.eval()

#Continual_Plot(args).plot_figures()
