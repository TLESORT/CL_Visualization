import os
import time
import torch
import numpy as np
import argparse
import datetime
import wandb

from Methods.trainer import Trainer
from utils import check_exp_config

parser = argparse.ArgumentParser()


# Algorithms Parameters
parser.add_argument('--name_algo', type=str,
                    choices=['baseline', 'rehearsal', 'ewc_diag', "ewc_diag_id", "ewc_kfac_id", 'ewc_kfac', 'ogd'],
                    default='baseline', help='Approach type')
parser.add_argument('--OutLayer', default="Linear", type=str,
                    choices=['Linear', 'CosLayer', "Linear_no_bias", 'MIMO_Linear', 'MIMO_Linear_no_bias',
                             'MIMO_CosLayer', 'MeanLayer', 'MedianLayer', 'KNN', 'SLDA', 'WeightNorm', 'OriginalWeightNorm'],
                    help='type of ouput layer used for the NN')
parser.add_argument('--pretrained_on', default=None, type=str,
                    choices=[None, "CIFAR10", "CIFAR100", "ImageNet"],
                    help='dataset source of a pretrained model')
parser.add_argument('--architecture', default="resnet", type=str,
                    choices=["resnet", "alexnet", "vgg", "googlenet"],
                    help='architecture')

# Logs / Data / Paths
parser.add_argument('--dataset', default="MNIST", type=str,
                    choices=['MNIST', 'mnist_fellowship', 'CIFAR10', 'CIFAR100', 'SVHN', 'Core50', 'ImageNet',
                             "Core10Lifelong", "Core10Mix"], help='dataset name')
parser.add_argument('--scenario_name', type=str, choices=['Disjoint', 'Rotations', 'Domain'], default="Disjoint",
                    help='continual scenario')
parser.add_argument('--num_tasks', type=int, default=5, help='Task number')
parser.add_argument('--root_dir', default="./Archives", type=str,
                    help='data directory name')
parser.add_argument('--data_dir', default="./Archives/Datasets", type=str,
                    help='data directory name')
parser.add_argument('--pmodel_dir', default="Pretrained", type=str,
                    help='data directory name')

# Model HPs
parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--importance', default=1.0, type=float, help='Importance of penalty')
parser.add_argument('--nb_epochs', default=5, type=int,
                    help='Epochs for each task')
parser.add_argument('--batch_size', default=264, type=int, help='batch size')
parser.add_argument('--masked_out', default=None, type=str, choices=[None, "single", "group", "multi-head"],
                    help='if single we only update one out dimension,'
                         ' if group mask the classes in the batch,'
                         ' multi-head is for training with multi head while testing single head')
parser.add_argument('--subset', type=int, default=None,
                    help='we can replace the full tasks by a subset of samples randomly selected')
parser.add_argument('--seed', default="1664", type=int,
                    help='seed for number generator')

# FLAGS
parser.add_argument('--finetuning', action='store_true', default=False,
                    help='decide if we finetune pretrained models')
parser.add_argument('--test_label', action='store_true', default=False,
                    help='define if we use task label at test')
parser.add_argument('--reset_opt', action='store_true', default=False,
                    help='reset opt at each new task')
parser.add_argument('--load_first_task', action='store_true', default=False, help='dev flag')
parser.add_argument('--no_train', action='store_true', default=False, help='flag to only analyse or plot figures')
parser.add_argument('--analysis', action='store_true', default=False, help='flag for analysis')
parser.add_argument('--fast', action='store_true', default=False, help='if fast we avoid most logging')
parser.add_argument('--dev', action='store_true', default=False, help='dev flag')
parser.add_argument('--verbose', action='store_true', default=False, help='dev flag')

config = parser.parse_args()
torch.manual_seed(config.seed)
np.random.seed(config.seed)

if config.seed == 0 or config.seed == 1664:
    task_order = np.arange(config.num_tasks)
else:
    task_order = np.random.permutation(config.num_tasks)

config.task_order = task_order
experiment_id = f"{config.dataset}"

config.pmodel_dir = os.path.join(config.root_dir, config.pmodel_dir)
if not os.path.exists(config.pmodel_dir):
    os.makedirs(config.pmodel_dir)

experiment_id = os.path.join(experiment_id, config.scenario_name, f"{config.num_tasks}-tasks")

if config.pretrained_on is not None:
    preposition = ''
    if config.finetuning:
        preposition = 'finetuning_'

    experiment_id = os.path.join(experiment_id, f"{preposition}pretrained_on_{config.pretrained_on}")

if config.subset is not None:
    experiment_id = os.path.join(experiment_id, f"subset-{config.subset}")
    if config.OutLayer in ['MeanLayer', 'MedianLayer', 'KNN', 'SLDA']:
        config.nb_epochs = 1  # for layer that does not learn there is not need for more than one epoch

if config.test_label:
    experiment_id = os.path.join(experiment_id, "MultiH")
else:
    experiment_id = os.path.join(experiment_id, "SingleH")

if config.masked_out == "single":
    name_out = f"{config.OutLayer}_Masked"
elif config.masked_out == "group":
    name_out = f"{config.OutLayer}_GMasked"
elif config.masked_out == "multi-head":
    name_out = f"{config.OutLayer}_Mhead"
elif config.masked_out == "right":
    name_out = f"{config.OutLayer}_RMasked"
else:
    name_out = config.OutLayer

experiment_id = os.path.join(experiment_id, f"seed-{config.seed}", name_out)

config.root_dir = os.path.join(config.root_dir, experiment_id)
if not os.path.exists(config.root_dir):
    os.makedirs(config.root_dir)

# save args parameters and date
if not config.no_train:
    file_name = os.path.join(config.root_dir, f"config_{config.name_algo}.txt")
    print(f"Save args in {file_name}")
    with open(file_name, 'w') as fp:
        fp.write(f'{datetime.datetime.now()} \n')
        fp.write(str(config).replace(",", ",\n"))

if config.subset is None:
    experiment_label = f"{config.dataset}-pretrained-{config.pretrained_on}"
else:
    experiment_label = f"{config.dataset}-pretrained-{config.pretrained_on}-subset-{config.subset}"

experiment_id = experiment_id.replace("/", "-")

if not config.dev:

    # Check if experience already exists
    exp_already_done = check_exp_config(config, name_out)
    if exp_already_done:
        print(f"This experience has already been run and finished: {experiment_id}")
        exit()
    else:
        print("This experience has not been run yet")

    for i in range(10):
        try:
            wandb.init(
                project="CL_Visualization", settings=wandb.Settings(start_method='fork'),
                group=experiment_label,
                id=experiment_id + '-' + wandb.util.generate_id(),
                entity='tlesort',
                notes=f"Experiment: Dataset {config.dataset}, OutLayer {config.OutLayer}, Pretrained on {config.pretrained_on}",
                tags=[config.dataset, config.OutLayer],
                config=config,
            )
            break
        except:
            print(f"Retrying {i}")
            time.sleep(10)

    wandb.config.update({"OutLayer": name_out}, allow_val_change=True)

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

    analysis_tool = Continual_Analysis(config)
    analysis_tool.analysis()

# if not (config.fast or config.dev):
#     from Plot.plot import Continual_Plot
#     Continual_Plot(config).plot_figures(method=config.name_algo)
