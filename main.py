import os
import shutil
import time
import torch
import numpy as np
import argparse
import datetime
import wandb
from typing import List

from Methods.trainer import Trainer
from utils import check_exp_config

parser = argparse.ArgumentParser()


# Algorithms Parameters
parser.add_argument('--name_algo', type=str,
                    choices=['baseline', 'rehearsal', 'ewc_diag', "ewc_diag_id", "ewc_kfac_id", 'ewc_kfac', 'ogd',
                             'erm', 'irm', 'ib_irm', 'ib_erm', 'SpectralDecoupling', 'GroupDRO'],
                    default='baseline', help='Approach type')
parser.add_argument('--scenario_name', type=str, choices=['Disjoint', 'Rotations', 'Domain', 'SpuriousFeatures'], default="Disjoint", help='continual scenario')
parser.add_argument('--OutLayer', default="Linear", type=str,
                    choices=['Linear', 'CosLayer', 'FCosLayer', "Linear_no_bias", 'MIMO_Linear', 'MIMO_Linear_no_bias',
                             'MIMO_CosLayer', 'MeanLayer', 'MedianLayer', 'KNN', 'SLDA', 'WeightNorm', 'OriginalWeightNorm'],
                    help='type of ouput layer used for the NN')
parser.add_argument('--pretrained_on', default=None, type=str,
                    choices=[None, "NA","CIFAR10", "CIFAR100", "ImageNet"],
                    help='dataset source of a pretrained model')
parser.add_argument('--architecture', default="resnet", type=str,
                    choices=["resnet", "alexnet", "vgg", "googlenet"],
                    help='architecture')

# Logs / Data / Paths
parser.add_argument('--dataset', default="MNIST", type=str,
                    choices=['MNIST', 'mnist_fellowship', 'CIFAR10', 'CIFAR100', 'SVHN', 'CUB200', 'AwA2','Core50', 'ImageNet',
                             "Core10Lifelong", "Core10Mix", 'CIFAR100Lifelong', "Tiny","OxfordPet", "OxfordFlower102",
                             "VLCS", "TerraIncognita"], help='dataset name')


parser.add_argument('--num_tasks', type=int, default=5, help='Task number')
parser.add_argument('--num_classes', type=int, default=-1, help='Num classes taken in the dataset')
parser.add_argument('--classes_per_tasks', type=int, default=-1, help='Num classes per tasks')
parser.add_argument('--spurious_corr', type=float, default=1.0, help='Correlation between the spurious features and the labels')
parser.add_argument('--support', type=float, default=1.0, help='amount of data of the original data in each task for spurious correlation scenarios')
parser.add_argument('--increments', type=int, nargs="*", default=[0], help='to manually set the number of increments.')
parser.add_argument('--root_dir', default="./Archives", type=str,
                    help='data directory name')
parser.add_argument('--data_dir', default="../Datasets", type=str,
                    help='data directory name')
parser.add_argument('--pmodel_dir', default="Pretrained", type=str,
                    help='data directory name')

# Model HPs
parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
parser.add_argument('--opt_name', default="SGD", type=str,
                    choices=['SGD', 'Adam'],
                    help='data directory name')
parser.add_argument('--weight_decay', default=0.0, type=float, help='weight_decay')
parser.add_argument('--irm_penalty_anneal_iters', default=500, type=float, help='irm_penalty_anneal_iters')
parser.add_argument('--ib_penalty_anneal_iters', default=500, type=float, help='ib_penalty_anneal_iters')
parser.add_argument('--irm_lambda', default=0.1, type=float, help='irm_lambda')
parser.add_argument('--ib_lambda', default=0.0, type=float, help='ib_lambda')
parser.add_argument('--groupdro_eta', default=1e-2, type=float, help='_hparam(\'groupdro_eta\', 1e-2, lambda r: 10**r.uniform(-3, -1))')
parser.add_argument('--sd_reg', default=0.1, type=float, help='0.1, lambda r: 10**r.uniform(-5, -1)')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--dropout', default=None, type=float, help='dropout between feature and head')
parser.add_argument('--importance', default=1.0, type=float, help='Importance of penalty')
parser.add_argument('--normalize', action="store_true", help="normalize the loss of irm / vrex")
parser.add_argument('--nb_epochs', default=5, type=int,
                    help='Epochs for each task')
parser.add_argument('--batch_size', default=256, type=int, help='batch size')
parser.add_argument('--nb_samples_rehearsal_per_class', default=100, type=int, help='nb_samples_rehearsal_per_class')
parser.add_argument('--replay_balance', default=1.0, type=float, help=' define how much replay we realize, 1 menas perfect balance between previous and past data.')
parser.add_argument('--masked_out', default=None, type=str, choices=[None, "None", "single", "group", "multi-head"],
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
parser.add_argument('--OOD_Training', action='store_true', default=False,
                    help='ood training all tasks are available at the same time as different envs')
parser.add_argument('--proj_drift_eval', action='store_true', default=False, help='eval the proj drift')
parser.add_argument('--keep_task_order', action='store_true', default=False, help='keep the task order')
parser.add_argument('--test_label', action='store_true', default=False, help='define if we use task label at test')
parser.add_argument('--reset_opt', action='store_true', default=False, help='reset opt at each new task')
parser.add_argument('--load_first_task', action='store_true', default=False, help='dev flag')
parser.add_argument('--no_train', action='store_true', default=False, help='flag to only analyse or plot figures')
parser.add_argument('--analysis', action='store_true', default=False, help='flag for analysis')
parser.add_argument('--fast', action='store_true', default=False, help='if fast we avoid most logging')
parser.add_argument('--sweep', action='store_true', default=False, help='if sweep we do not check if exps has been already ran')
parser.add_argument('--sweeps_HPs', action='store_true', default=False, help='use HPs previously got by sweep runnig')
parser.add_argument('--project_name', default="CLOOD", type=str, help='project name for wandb')
parser.add_argument('--offline', action='store_true', default=False, help='does not save in wandb')
parser.add_argument('--offline_wandb', action='store_true', default=False, help='does save in wandb but offline')
parser.add_argument('--dev', action='store_true', default=False, help='dev flag')
parser.add_argument('--verbose', action='store_true', default=False, help='dev flag')

config = parser.parse_args()
torch.manual_seed(config.seed)
np.random.seed(config.seed)


if config.classes_per_tasks != -1 and config.num_classes == -1:
    config.num_classes = config.classes_per_tasks * config.num_tasks


if config.dataset in ["OxfordPet", "OxfordFlower102"] and config.scenario_name=="Disjoint" and config.increments[0]==0:
    if config.dataset == "OxfordPet":
        nb_classes = 37
    elif config.dataset == "OxfordFlower102":
        nb_classes = 102
    class_per_task = nb_classes // config.num_tasks
    remaining = nb_classes % config.num_tasks
    config.increments = [class_per_task] * (config.num_tasks-1) + [class_per_task+remaining]
    print(config.increments)

if config.masked_out == "None":
    config.masked_out = None

if config.sweeps_HPs:
    if config.project_name == "CLOOD":
        from Reproducibility.SpuriousFeatures.HPs import get_selected_HPs_Spurious
        config = get_selected_HPs_Spurious(config)

# cluster sweep
slurm_tmpdir = os.environ.get('SLURM_TMPDIR')
if not (slurm_tmpdir is None):
    config.root_dir = os.path.join(slurm_tmpdir, "Archives")
    config.data_dir = os.path.join(slurm_tmpdir, "Datasets")
    if not os.path.exists(config.data_dir):
        os.makedirs(config.data_dir)

    data_storage_dir = "/network/projects/_groups/large_cl/Datasets/"
    second_data_storage_dir = "./Archives/Datasets/"

    #['MNIST', 'mnist_fellowship', 'CIFAR10', 'CIFAR100', 'SVHN', 'CUB200', 'AwA2','Core50', 'ImageNet',
     # "Core10Lifelong", "Core10Mix", 'CIFAR100Lifelong', "Tiny","OxfordPet", "OxfordFlower102",
     # "VLCS", "TerraIncognita"]
    list_files = []
    if config.dataset == "GTSRB":
        # files: GT-final_test.csv  GTSRB_Final_Test_GT.zip GTSRB_Final_Test_Images.zip
        list_files = ["GT-final_test.csv", "GTSRB_Final_Test_GT.zip", "GTSRB_Final_Test_Images.zip"]
    elif config.dataset == "VLCS":
        list_files = ["VLCS.tar.gz"]
    elif config.dataset == "TerraIncognita":
        list_files = ["eccv_18_all_images_sm.tar.gz", "caltech_camera_traps.json.zip"]
    elif config.dataset == "CUB200":
        list_files = ["CUB_200_2011.tgz"]
    elif config.dataset == "Tiny":
        list_files = ["tiny-imagenet-200.zip"]
    elif config.dataset == "CIFAR100":
        list_files = ["cifar-100-python.tar.gz"]
    elif config.dataset == "CIFAR10":
        list_files = ["cifar-10-python.tar.gz"]

    for filename in list_files:
        full_path = os.path.join(config.data_dir, filename)
        if not os.path.exists(full_path):
            storage_path = os.path.join(data_storage_dir, filename)
            second_storage_path = os.path.join(second_data_storage_dir, filename)
            if os.path.exists(storage_path):
                shutil.copy(storage_path, full_path)  # shutil.copy(src_path, dst_path)
            elif os.path.exists(second_storage_path):
                shutil.copy(second_storage_path, full_path)  # shutil.copy(src_path, dst_path)
            else:
                pass  # file will be downloaded automatically then...

config.original_root = config.root_dir

if config.seed == 0 or config.seed == 1664:
    task_order = np.arange(config.num_tasks)
else:
    task_order = np.random.permutation(config.num_tasks)

config.task_order = task_order
experiment_id = f"{config.dataset}"

config.pmodel_dir = os.path.join(config.root_dir, config.pmodel_dir)
if not os.path.exists(config.pmodel_dir):
    os.makedirs(config.pmodel_dir)

if not os.path.exists(config.data_dir):
    os.makedirs(config.data_dir)

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

config.log_dir = os.path.join(config.root_dir, "Logs", config.scenario_name)
if not os.path.exists(config.log_dir):
    os.makedirs(config.log_dir)
config.sample_dir = os.path.join(config.root_dir, "Samples")
if not os.path.exists(config.sample_dir):
    os.makedirs(config.sample_dir)

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

if config.pretrained_on == "NA": config.pretrained_on = None

if not (config.dev or config.offline):

    # Check if experience already exists
    # exp_already_done=False
    # if config.seed != 1664 and not config.sweep: # this seed is vip and sweep already check if exps are already run
    #     exp_already_done = check_exp_config(config, name_out)
    # if exp_already_done:
    #     print(f"This experience has already been run and finished: {experiment_id}")
    #     exit()
    # else:
    #     print("This experience has not been run yet")

    for i in range(10):
        try:
            if config.offline_wandb:
                os.environ["WANDB_MODE"] = "offline"
                # to synchronize : run in terminal wandb sync YOUR_RUN_DIRECTORY
                # ex : wandb sync wandb/offline-run-20210903_135922-MNIST-class_inc-AverageHash-seed_1665-1468z5g8/
            else:
                os.environ["WANDB_MODE"] = "online"

            wandb.init(
                dir=config.original_root,
                project=config.project_name, settings=wandb.Settings(start_method='fork'),
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

    if config.masked_out is not None:
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

elif config.name_algo == "irm":
    from Methods.IRM import IRM
    Algo = IRM(config)

elif config.name_algo == "erm":
    from Methods.IRM import ERM
    Algo = ERM(config)
elif config.name_algo == "ib_erm":
    from Methods.IRM import IBERM
    Algo = IBERM(config)
elif config.name_algo == "ib_irm":
    from Methods.IRM import IBIRM
    Algo = IBIRM(config)
elif config.name_algo == "SpectralDecoupling":
    from Methods.IRM import SpectralDecoupling
    Algo = SpectralDecoupling(config)
elif config.name_algo == "GroupDRO":
    from Methods.IRM import GroupDRO
    Algo = GroupDRO(config)
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
