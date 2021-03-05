
import os
import torch
import argparse

from trainer import Trainer

# le gradient depend de la couche de sortie (pas toujours)
# mais surtout de la loss function
# c'est l'occasion de trouver des loss / couche de sortie plus adaptées aux continual learning

parser = argparse.ArgumentParser()
parser.add_argument('--name_algo', type=str,
                    choices=['baseline', 'rehearsal', 'ewc_diag', "ewc_diag_id", "ewc_kfac_id", 'ewc_kfac'],
                    default='baseline', help='Approach type')
parser.add_argument('--scenario_name', type=str, choices=['Disjoint', 'Rotations', 'Domain'], default="Disjoint",
                    help='continual scenario')
parser.add_argument('--num_tasks', type=int, default=5, help='Task number')
parser.add_argument('--root_dir', default="./Archives", type=str,
                    help='data directory name')
parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
parser.add_argument('--importance', default=1.0, type=float, help='Importance of penalty')
parser.add_argument('--nb_epochs', default=5, type=int,
                    help='Epochs for each task')
parser.add_argument('--batch_size', default=264, type=int, help='batch size')
parser.add_argument('--test_label', action='store_true', default=False,
                    help='define if we use task label at test')
parser.add_argument('--expert', action='store_true', default=False,
                    help='define if we use expert model who has access to all data')
parser.add_argument('--no_train', action='store_true', default=False, help='dev flag')
parser.add_argument('--dev', action='store_true', default=False, help='dev flag')
parser.add_argument('--verbose', action='store_true', default=False, help='dev flag')
parser.add_argument('--dataset', default="MNIST", type=str,
                    choices=['MNIST','mnist_fellowship'], help='dataset name')
parser.add_argument('--seed', default="1992", type=int,
                    help='seed for number generator')

args = parser.parse_args()
torch.manual_seed(args.seed)

args.root_dir = os.path.join(args.root_dir, args.dataset)
if not os.path.exists(args.root_dir):
    os.makedirs(args.root_dir)

if args.name_algo == "baseline":
    Algo = Trainer(args, args.root_dir, args.scenario_name, args.num_tasks, args.verbose, args.dev)
elif args.name_algo == "rehearsal":
    from rehearsal import Rehearsal
    Algo = Rehearsal(args, args.root_dir, args.scenario_name, args.num_tasks, args.verbose, args.dev)
elif args.name_algo == "ewc_diag":
    from Ewc import EWC_Diag
    Algo = EWC_Diag(args, args.root_dir, args.scenario_name, args.num_tasks, args.verbose, args.dev)
elif args.name_algo == "ewc_diag_id":
    from Ewc import EWC_Diag_id
    Algo = EWC_Diag_id(args, args.root_dir, args.scenario_name, args.num_tasks, args.verbose, args.dev)
elif args.name_algo == "ewc_kfac_id":
    from Ewc import EWC_KFAC_id
    Algo = EWC_KFAC_id(args, args.root_dir, args.scenario_name, args.num_tasks, args.verbose, args.dev)
elif args.name_algo == "ewc_kfac":
    from Ewc import EWC_KFAC
    Algo = EWC_KFAC(args, args.root_dir, args.scenario_name, args.num_tasks, args.verbose, args.dev)
else:
    print("wrong name")


# Algo.eval()

if not args.no_train:
    Algo.continual_training()

from plot import Continual_Plot
Continual_Plot(args).plot_figures(method=args.name_algo)