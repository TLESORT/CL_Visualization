import torch

from Models.model import Model

import torchvision.transforms as trsf


def get_scenario(dataset, scenario_name, nb_tasks, transform=None):
    if scenario_name == "Rotations":
        from continuum import Rotations
        scenario = Rotations(dataset, nb_tasks=nb_tasks, transformations=transform)
    elif scenario_name == "Disjoint":
        from continuum import ClassIncremental
        scenario = ClassIncremental(dataset, nb_tasks=nb_tasks, transformations=transform)
    elif scenario_name == "Domain":
        from continuum import InstanceIncremental
        scenario = InstanceIncremental(dataset, nb_tasks=nb_tasks, transformations=transform)

    return scenario


def get_dataset(path_dir, name_dataset, name_scenario, train="True"):
    if name_dataset == "MNIST":
        from continuum.datasets import MNIST
        dataset = MNIST(path_dir, download=True, train=train)
    if name_dataset == "Core50":
        from continuum.datasets import Core50
        dataset = Core50(path_dir, download=False, train=train)
    elif name_dataset == "CIFAR10":
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
        print("Dataset unKnown")
    return dataset


def get_transform(name_dataset, train="True"):
    if name_dataset == "Core50":
        normalize = trsf.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        resize = trsf.Resize(size=224)
        transform = trsf.Compose([resize, trsf.ToTensor(), normalize])
    elif name_dataset == "CIFAR10":
        transform = trsf.Compose([
            trsf.ToTensor(),
            trsf.Normalize(
                mean=[0.4914, 0.4822, 0.4465],  # mean=[0.5071, 0.4865, 0.4409] for cifar100
                std=[0.2023, 0.1994, 0.2010],  # std=[0.2009, 0.1984, 0.2023] for cifar100
            ),
        ])
    elif name_dataset == "CIFAR100":
        transform = trsf.Compose([
            trsf.ToTensor(),
            trsf.Normalize(
                mean=[0.5071, 0.4865, 0.4409],
                std=[0.2009, 0.1984, 0.2023],
            ),
        ])
    else:
        transform = None
    return [transform]


def get_model(name_dataset, scenario, pretrained_on, test_label, OutLayer, method):
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

        if name_dataset == "CIFAR10" or name_dataset == "CIFAR100" or name_dataset == "SVHN":
            from Models.cifar_models import CIFARModel
            model = CIFARModel(num_classes=10, OutLayer=OutLayer, pretrained_on=pretrained_on)

        elif name_dataset == "Core50":
            from Models.imagenet import ImageNetModel
            model = ImageNetModel(num_classes=10, OutLayer=OutLayer, pretrained=pretrained_on == "ImageNet",
                                  name_model="alexnet")
        else:
            model = Model(num_classes=scenario.nb_classes, OutLayer=OutLayer, pretrained_on=pretrained_on)

    return model.cuda()
