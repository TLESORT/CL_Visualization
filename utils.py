import torch
import numpy as np
import torchvision.transforms as trsf

from Models.model import Model

import torchvision.transforms as trsf
import torch.optim as optim
import numpy as np

from continuum.datasets import InMemoryDataset
from Models.model import Model

def get_lifelong_cifar100(dataset):
    x, y, _ = dataset.get_data()
    cifar100_coarse_labels = np.array([4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                                       3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                                       6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                                       0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                                       5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                                       16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                                       10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                                       2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                                       16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                                       18, 1, 2, 15, 6, 0, 17, 8, 14, 13])

    # they should be 20 corase labels with 5 classes each
    coarse_labels = np.unique(cifar100_coarse_labels)
    assert len(coarse_labels) == 20

    # we should have a 5*20 matrix with indexes of classes of each coarse_labels
    np_indexes_coarse_labels = np.zeros((5, 20))
    for i in range(20):
        indexes_coarse_labels = np.where(cifar100_coarse_labels == i)[0]
        assert len(indexes_coarse_labels) == 5, print(f"len(indexes_coarse_labels): {len(indexes_coarse_labels)}")
        np_indexes_coarse_labels[:, i] = np.array(indexes_coarse_labels)

    np_indexes_coarse_labels = np_indexes_coarse_labels.astype(int)

    # we can have 5 tasks with 1 classes per coarse labels to make a lifelong scenario

    # so we create the task label vector
    t = np.zeros(len(y))
    for i in range(5):
        indexes = np_indexes_coarse_labels[i,:]
        assert len(np.unique(cifar100_coarse_labels[indexes])) == 20, print(cifar100_coarse_labels[indexes])
        assert len(indexes) == 20, print(f"len(indexes) {len(indexes)}")
        for index in indexes:
            data_index_class = np.where(y==index)[0]
            t[data_index_class] = i

    # now we have t for each data point, we convert y into coarse labels
    y = cifar100_coarse_labels[y]
    assert len(y) == len(t)
    assert y.max() == 19
    assert len(np.unique(t)) == 5


    return InMemoryDataset(x, y.astype(int), t.astype(int), data_type="image_array")

def get_scenario(dataset, scenario_name, nb_tasks, increments=[0], transform=None, config=None):
    if scenario_name == "Rotations":
        from continuum import Rotations
        scenario = Rotations(dataset, nb_tasks=nb_tasks, transformations=transform)
    elif scenario_name == "Disjoint":
        from continuum import ClassIncremental
        if increments[0] == 0:
            scenario = ClassIncremental(dataset, nb_tasks=nb_tasks, transformations=transform)
        else:
            scenario = ClassIncremental(dataset, increment=increments, transformations=transform)
    elif scenario_name == "Domain":
        from continuum import ContinualScenario
        scenario = ContinualScenario(dataset, transformations=transform)
    elif scenario_name == "SpuriousFeatures":
        from scenario.spurious_features import SpuriousFeatures
        scenario = SpuriousFeatures(dataset, nb_tasks=nb_tasks, base_transformations=transform,
                                    correlation=config.spurious_corr, train=dataset.train, support=config.support)

    return scenario

def get_optim(name_optim, parameters, lr, momentum):
    if name_optim=="SGD":
        opt = optim.SGD(params=parameters, lr=lr, momentum=momentum)
    elif name_optim=="Adam":
        opt = optim.Adam(params=parameters, lr=lr)
    else:
        raise NotImplementedError("this opt is not implemented here")
    return opt

def get_dataset(path_dir, name_dataset, name_scenario, train="True"):
    if name_dataset == "MNIST":
        from continuum.datasets import MNIST
        dataset = MNIST(path_dir, download=True, train=train)
    # elif "SQOLOR" in name_dataset:
    #     dataset = generate_SQOLOR(path_dir, name_dataset, name_scenario, train=train)
    elif name_dataset == "Core50":
        from continuum.datasets import Core50
        dataset = Core50(path_dir, download=False, train=train)
    elif name_dataset == "Tiny":
        from continuum.datasets import TinyImageNet200
        dataset = TinyImageNet200(path_dir, download=True, train=train)
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
    elif name_dataset == "CIFAR100Lifelong":
        from continuum.datasets import CIFAR100
        dataset = CIFAR100(path_dir,
                           download=True,
                           labels_type="category",
                           task_labels="lifelong",
                           train=train, )
    elif name_dataset == "ImageNet":
        from continuum.datasets import ImageNet
        dataset = ImageNet(path_dir, download=True, train=train)
    elif name_dataset == "CUB200":
        from continuum.datasets import CUB200
        dataset = CUB200(path_dir, download=True, train=train)
    elif name_dataset == "AwA2":
        from continuum.datasets import AwA2
        dataset = AwA2(path_dir, download=True, train=train)
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
    if name_dataset in ["Core50", "Core10Lifelong", "Core10Mix", 'CUB200', 'AwA2', "Tiny"]:
        normalize = trsf.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        resize = trsf.Resize(size=(224, 224))
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
    elif name_dataset in ["CIFAR100", "CIFAR100Lifelong"]:
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
              architecture="resnet", dropout=None):
    if test_label:
        # there are no test label for domain incremental since the classes should be always the same
        # assert name_dataset == "Disjoint"

        list_classes_per_tasks = []
        for task_set in scenario:
            classes = task_set.get_classes()
            list_classes_per_tasks.append(classes)

    else:
        list_classes_per_tasks = None

    if name_dataset in ["CIFAR10", "CIFAR100", "SVHN", "CIFAR100Lifelong"]:
        from Models.cifar_models import CIFARModel
        model = CIFARModel(num_classes=scenario.nb_classes,
                           classes_per_head=list_classes_per_tasks,
                           OutLayer=OutLayer,
                           pretrained_on=pretrained_on,
                           model_dir=model_dir, dropout=dropout)

    elif name_dataset in ["Core50", "Core10Lifelong", "Core10Mix", 'CUB200', 'AwA2', "Tiny"]:
        from Models.imagenet import ImageNetModel
        model = ImageNetModel(num_classes=scenario.nb_classes,
                              classes_per_head=list_classes_per_tasks,
                              OutLayer=OutLayer,
                              pretrained=pretrained_on == "ImageNet",
                              name_model=architecture)
    elif type(scenario).__name__ == "SpuriousFeatures":
        model = Model(num_classes=scenario.nb_classes, OutLayer=OutLayer, pretrained_on=pretrained_on, input_dim=3)
    else:
        model = Model(num_classes=scenario.nb_classes,
                      classes_per_head=list_classes_per_tasks,
                      OutLayer=OutLayer,
                      method=method,
                      pretrained_on=pretrained_on)

    return model.cuda()


import wandb
from Plot.utils_wandb import select_run


def check_exp_config(config, name_out):
    api = wandb.Api()
    runs = api.runs(f"tlesort/{config.project_name}")
    exp_already_done = False

    for run in runs:
        if run.state == "finished":
            dict_config = {k: v for k, v in run.config.items() if not k.startswith('_')}
            exp_already_done = select_run(dict_config,
                                          config.scenario_name,
                                          config.name_algo,
                                          config.dataset,
                                          config.pretrained_on,
                                          config.num_tasks,
                                          name_out,
                                          config.subset,
                                          config.seed,
                                          config.lr,
                                          config.architecture,
                                          config.finetuning,
                                          config.test_label,
                                          config.spurious_corr,
                                          config.nb_samples_rehearsal_per_class)
            if exp_already_done:
                print(f"This experience has already be run and finished: {run.name}")
                print(dict_config)
                break
    return exp_already_done




