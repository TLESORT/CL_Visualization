import torch
import numpy as np
import torchvision.transforms as trsf

from Models.model import Model


def get_scenario(dataset, scenario_name, nb_tasks, transform=None, train=True):
    if scenario_name == "Rotations":
        from continuum import Rotations
        scenario = Rotations(dataset, nb_tasks=nb_tasks, transformations=transform)
    elif scenario_name == "Disjoint":
        from continuum import ClassIncremental
        scenario = ClassIncremental(dataset, nb_tasks=nb_tasks, transformations=transform)
    elif scenario_name == "Domain":
        from continuum import ContinualScenario
        scenario = ContinualScenario(dataset, transformations=transform)
    elif scenario_name == "SpuriousFeatures":
        from scenario.spurious_features import SpuriousFeatures
        scenario = SpuriousFeatures(dataset, nb_tasks=nb_tasks, base_transformations=transform, train=train)

    return scenario


def get_dataset(path_dir, name_dataset, name_scenario, train="True"):
    if name_dataset == "MNIST":
        from continuum.datasets import MNIST
        dataset = MNIST(path_dir, download=True, train=train)
    # elif "SQOLOR" in name_dataset:
    #     dataset = generate_SQOLOR(path_dir, name_dataset, name_scenario, train=train)
    elif name_dataset == "Core50":
        from continuum.datasets import Core50
        dataset = Core50(path_dir, download=False, train=train)
    elif name_dataset == "Core10Lifelong":
        from continuum.datasets import Core50
        dataset = Core50(path_dir, scenario="domains", classification="category", train=train)
    elif name_dataset == "Core10Mix":
        from continuum.datasets import Core50
        dataset = Core50(path_dir, scenario="objects", classification="category", train=train)
    elif name_dataset == "CIFAR10":
        from continuum.datasets import CIFAR10
        dataset = CIFAR10(path_dir, download=True, train=train)
    elif name_dataset == "CIFAR100":
        from continuum.datasets import CIFAR100
        dataset = CIFAR100(path_dir, download=True, train=train)
    elif name_dataset == "ImageNet":
        from continuum.datasets import CIFAR10
        dataset = CIFAR10(path_dir, download=True, train=train)
    elif name_dataset == "SVHN":
        from torchvision.datasets import SVHN
        from continuum.datasets import PyTorchDataset
        dataset = PyTorchDataset("path_dir", dataset_type=SVHN, train=True, download=True)
    elif name_dataset == "mnist_fellowship" and name_scenario == "Disjoint":
        from continuum.datasets import MNISTFellowship
        dataset = MNISTFellowship(path_dir, download=True, train=train)
    elif name_dataset == "mnist_fellowship" and name_scenario == "Domain":
        dataset = MNISTFellowship(path_dir, download=True, train=train, update_labels=False)
    else:
        raise NotImplementedError(f"Dataset {name_dataset} Unknown")
    return dataset


def get_transform(name_dataset, architecture, train="True"):
    list_transform = None
    if name_dataset in ["Core50", "Core10Lifelong", "Core10Mix"]:
        normalize = trsf.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        resize = trsf.Resize(size=224)
        transform = trsf.Compose([resize, trsf.ToTensor(), normalize])
        list_transform = [transform]
    elif name_dataset == "CIFAR10":
        transform = trsf.Compose([
            trsf.ToTensor(),
            trsf.Normalize(
                mean=[0.4914, 0.4822, 0.4465],  # mean=[0.5071, 0.4865, 0.4409] for cifar100
                std=[0.2023, 0.1994, 0.2010],  # std=[0.2009, 0.1984, 0.2023] for cifar100
            ),
        ])
        list_transform = [transform]
    elif name_dataset == "CIFAR100":
        transform = trsf.Compose([
            trsf.ToTensor(),
            trsf.Normalize(
                mean=[0.5071, 0.4865, 0.4409],
                std=[0.2009, 0.1984, 0.2023],
            ),
        ])
        list_transform = [transform]

    return list_transform


def get_model(name_dataset, scenario, pretrained_on, test_label, OutLayer, method, model_dir=None,
              architecture="resnet"):
    if test_label:
        # there are no test label for domain incremental since the classes should be always the same
        # assert name_dataset == "Disjoint"

        list_classes_per_tasks = []
        for task_set in scenario:
            classes = task_set.get_classes()
            list_classes_per_tasks.append(classes)

        model = Model(num_classes=scenario.nb_classes,
                      classes_per_head=list_classes_per_tasks,
                      OutLayer=OutLayer,
                      method=method)
    else:

        if name_dataset in ["CIFAR10", "CIFAR100", "SVHN"]:
            from Models.cifar_models import CIFARModel
            model = CIFARModel(num_classes=scenario.nb_classes, OutLayer=OutLayer, pretrained_on=pretrained_on,
                               model_dir=model_dir)
            print("double check")

        elif name_dataset in ["Core50", "Core10Lifelong", "Core10Mix"]:
            from Models.imagenet import ImageNetModel
            model = ImageNetModel(num_classes=scenario.nb_classes, OutLayer=OutLayer,
                                  pretrained=pretrained_on == "ImageNet",
                                  name_model=architecture)
        elif type(scenario).__name__ == "SpuriousFeatures":
            model = Model(num_classes=scenario.nb_classes, OutLayer=OutLayer, pretrained_on=pretrained_on, input_dim=3)
        else:
            model = Model(num_classes=scenario.nb_classes, OutLayer=OutLayer, pretrained_on=pretrained_on)

    return model.cuda()


import wandb
from Plot.utils_wandb import select_run


def check_exp_config(config, name_out):
    api = wandb.Api()
    runs = api.runs("tlesort/CL_Visualization")
    exp_already_done = False

    for run in runs:
        if run.state == "finished":
            dict_config = {k: v for k, v in run.config.items() if not k.startswith('_')}
            exp_already_done = select_run(dict_config,
                                          config.dataset,
                                          config.pretrained_on,
                                          config.num_tasks,
                                          name_out,
                                          config.subset,
                                          config.seed,
                                          config.lr,
                                          config.architecture)
            if exp_already_done:
                print(f"This experience has already be run and finished: {run.name}")
                print(dict_config)
                break
    return exp_already_done
