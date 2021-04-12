
import os
import torch
import numpy as np
import argparse
import datetime

from Methods.trainer import Trainer

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
parser.add_argument('--masked_out', action='store_true', default=False, help='if true we only update one out dimension')
parser.add_argument('--OutLayer', default="Linear", type=str,
                    choices=['Linear', 'CosLayer', 'SLDA'],
                    help='type of ouput layer used for the NN')
parser.add_argument('--pretrained_on', default="None", type=str,
                    choices=[None, "CIFAR10", "CIFAR100"],
                    help='dataset source of a pretrained model')
parser.add_argument('--load_first_task', action='store_true', default=False, help='dev flag')
parser.add_argument('--no_train', action='store_true', default=False, help='dev flag')
parser.add_argument('--analysis', action='store_true', default=False, help='dev flag')
parser.add_argument('--fast', action='store_true', default=False, help='if fast we avoid most logging')
parser.add_argument('--dev', action='store_true', default=False, help='dev flag')
parser.add_argument('--verbose', action='store_true', default=False, help='dev flag')
parser.add_argument('--dataset', default="MNIST", type=str,
                    choices=['MNIST','mnist_fellowship', 'CIFAR10', 'CIFAR100', 'SVHN'], help='dataset name')
parser.add_argument('--seed', default="1664", type=int,
                    help='seed for number generator')

config = parser.parse_args()
torch.manual_seed(config.seed)
np.random.seed(config.seed)

config.data_dir = os.path.join(config.root_dir, "Datasets")
config.root_dir = os.path.join(config.root_dir, config.dataset)
if config.test_label:
    config.root_dir = os.path.join(config.root_dir, "MultiH")
else:
    config.root_dir = os.path.join(config.root_dir, "SingleH")
config.root_dir = os.path.join(config.root_dir, f"seed-{config.seed}")

if not os.path.exists(config.root_dir):
    os.makedirs(config.root_dir)


# save args parameters and date
if not config.no_train:
    file_name = os.path.join(config.root_dir, f"config_{config.name_algo}.txt")
    print(f"Save args in {file_name}")
    with open(file_name, 'w') as fp:
        fp.write(f'{datetime.datetime.now()} \n')
        fp.write(str(config).replace(",",",\n"))

if config.name_algo == "baseline":
    Algo = Trainer(config)
elif config.name_algo == "rehearsal":
    from Methods.rehearsal import Rehearsal
    Algo = Rehearsal(config)
elif config.name_algo == "ewc_diag":
    from Methods.Ewc import EWC_Diag
    Algo = EWC_Diag(config)
elif config.name_algo == "ewc_diag_id":
    from Methods.Ewc import EWC_Diag_id
    Algo = EWC_Diag_id(config)
elif config.name_algo == "ewc_kfac_id":
    from Methods.Ewc import EWC_KFAC_id
    Algo = EWC_KFAC_id(config)
elif config.name_algo == "ewc_kfac":
    from Methods.Ewc import EWC_KFAC
    Algo = EWC_KFAC(config)
elif config.name_algo == "ogd":
    from Methods.OGD import OGD
    Algo = OGD(config)
else:
    print("wrong name")

print("*********  START TRAINING *********")
print(config)

# Algo.eval()

if not config.no_train:
    Algo.continual_training()

if config.analysis:
    from Plot.Analysis import Continual_Analysis
    analysis_tool=Continual_Analysis(config)
    analysis_tool.analysis()

if not (config.fast or config.dev):
    from Plot.plot import Continual_Plot
    Continual_Plot(config).plot_figures(method=config.name_algo)