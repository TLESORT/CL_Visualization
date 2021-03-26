
import os
import torch
import numpy as np
import argparse
import datetime

from Methods.trainer import Trainer

# le gradient depend de la couche de sortie (pas toujours)
# mais surtout de la loss function
# c'est l'occasion de trouver des loss / couche de sortie plus adaptées aux continual learning

parser = argparse.ArgumentParser()
parser.add_argument('--name_algo', type=str,
                    choices=['baseline', 'rehearsal', 'ewc_diag', "ewc_diag_id", "ewc_kfac_id", 'ewc_kfac', 'ogd'],
                    default='baseline', help='Approach type')
parser.add_argument('--scenario_name', type=str, choices=['Disjoint', 'Rotations', 'Domain'], default="Disjoint",
                    help='continual scenario')
parser.add_argument('--num_tasks', type=int, default=5, help='Task number')
parser.add_argument('--root_dir', default="./Archives", type=str,
                    help='data directory name')
parser.add_argument('--data_dir', default="./Archives/Datasets", type=str,
                    help='data directory name')
parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--importance', default=1.0, type=float, help='Importance of penalty')
parser.add_argument('--nb_epochs', default=5, type=int,
                    help='Epochs for each task')
parser.add_argument('--batch_size', default=264, type=int, help='batch size')
parser.add_argument('--test_label', action='store_true', default=False,
                    help='define if we use task label at test')
parser.add_argument('--expert', action='store_true', default=False,
                    help='define if we use expert model who has access to all data')
parser.add_argument('--masked_out', action='store_true', default=False, help='if true we only update one out dimension')
parser.add_argument('--cosLayer', action='store_true', default=False, help='test the new layer')
parser.add_argument('--pretrained', action='store_true', default=False, help='dev flag')
parser.add_argument('--no_train', action='store_true', default=False, help='dev flag')
parser.add_argument('--fast', action='store_true', default=False, help='if fast we avoid most logging')
parser.add_argument('--dev', action='store_true', default=False, help='dev flag')
parser.add_argument('--verbose', action='store_true', default=False, help='dev flag')
parser.add_argument('--dataset', default="MNIST", type=str,
                    choices=['MNIST','mnist_fellowship', 'CIFAR10', 'CIFAR100', 'SVHN'], help='dataset name')
parser.add_argument('--seed', default="1664", type=int,
                    help='seed for number generator')

args = parser.parse_args()
torch.manual_seed(args.seed)
np.random.seed(args.seed)

args.data_dir = os.path.join(args.root_dir, "Datasets")
args.root_dir = os.path.join(args.root_dir, args.dataset)
if args.test_label:
    args.root_dir = os.path.join(args.root_dir, "MultiH")
else:
    args.root_dir = os.path.join(args.root_dir, "SingleH")
args.root_dir = os.path.join(args.root_dir, f"seed-{args.seed}")

if not os.path.exists(args.root_dir):
    os.makedirs(args.root_dir)


# save args parameters and date
file_name = os.path.join(args.root_dir, f"args.txt")
with open(file_name, 'w') as fp:
    fp.write(f'{datetime.datetime.now()} \n')
    fp.write(str(args).replace(",",",\n"))

if args.name_algo == "baseline":
    Algo = Trainer(args, args.root_dir, args.scenario_name, args.num_tasks, args.verbose, args.dev)
elif args.name_algo == "rehearsal":
    from Methods.rehearsal import Rehearsal
    Algo = Rehearsal(args, args.root_dir, args.scenario_name, args.num_tasks, args.verbose, args.dev)
elif args.name_algo == "ewc_diag":
    from Methods.Ewc import EWC_Diag
    Algo = EWC_Diag(args, args.root_dir, args.scenario_name, args.num_tasks, args.verbose, args.dev)
elif args.name_algo == "ewc_diag_id":
    from Methods.Ewc import EWC_Diag_id
    Algo = EWC_Diag_id(args, args.root_dir, args.scenario_name, args.num_tasks, args.verbose, args.dev)
elif args.name_algo == "ewc_kfac_id":
    from Methods.Ewc import EWC_KFAC_id
    Algo = EWC_KFAC_id(args, args.root_dir, args.scenario_name, args.num_tasks, args.verbose, args.dev)
elif args.name_algo == "ewc_kfac":
    from Methods.Ewc import EWC_KFAC
    Algo = EWC_KFAC(args, args.root_dir, args.scenario_name, args.num_tasks, args.verbose, args.dev)
elif args.name_algo == "ogd":
    from Methods.OGD import OGD
    Algo = OGD(args, args.root_dir, args.scenario_name, args.num_tasks, args.verbose, args.dev)
else:
    print("wrong name")

print("*********  START TRAINING *********")
print(args)

# Algo.eval()

if not args.no_train:
    Algo.continual_training()

if not args.fast:
    from Plot.plot import Continual_Plot
    Continual_Plot(args).plot_figures(method=args.name_algo)