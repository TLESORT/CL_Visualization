
import torch

from Models.model import Model
from Models.resnet import cifar_resnet20


def get_scenario(dataset, scenario_name, nb_tasks):
    if scenario_name == "Rotations":
        from continuum import Rotations
        scenario = Rotations(dataset, nb_tasks=nb_tasks)
    elif scenario_name == "Disjoint":
        from continuum import ClassIncremental
        scenario = ClassIncremental(dataset, nb_tasks=nb_tasks)
    elif scenario_name == "Domain":
        from continuum import InstanceIncremental
        scenario = InstanceIncremental(dataset, nb_tasks=nb_tasks)

    return scenario

def get_dataset(path_dir, name_dataset, name_scenario, train="True"):
    if name_dataset == "MNIST":
        from continuum.datasets import MNIST
        dataset = MNIST(path_dir, download=True, train=train)
    elif name_dataset == "CIFAR10":
        from continuum.datasets import CIFAR10
        dataset = CIFAR10(path_dir, download=True, train=train)
    elif name_dataset == "SVHN":
        from torchvision.datasets import SVHN
        from continuum.datasets import PyTorchDataset
        dataset = PyTorchDataset("path_dir", dataset_type=SVHN, train=True, download=True)
    elif name_dataset == "mnist_fellowship" and name_scenario=="Disjoint":
        from continuum.datasets import MNISTFellowship
        dataset = MNISTFellowship(path_dir, download=True, train=train)
    elif name_dataset == "mnist_fellowship" and name_scenario=="Domain":
        dataset = MNISTFellowship(path_dir, download=True, train=train, update_labels=False)
    else:
        print("Dataset unKnown")
    return dataset



def get_model(name_dataset, scenario, pretrained_on, test_label, OutLayer, method):

    if test_label:
        # there are no test label for domain incremental since the classes should be always the same
        #assert name_dataset == "Disjoint"

        list_classes_per_tasks = []
        for task_set in scenario:
            classes = task_set.get_classes()
            list_classes_per_tasks.append(classes)

        model = Model(num_classes=scenario.nb_classes,
                    classes_per_head=list_classes_per_tasks,
                    OutLayer=OutLayer,
                    method=method).cuda()
    else:

        if name_dataset == "CIFAR10" or name_dataset == "CIFAR100" or name_dataset == "SVHN":

            if pretrained_on is not None:
                model = cifar_resnet20(pretrained=pretrained_on)
                model.num_classes = 10 # manual correction
                for param in model.parameters():
                    param.requires_grad = False
            else:
                model = cifar_resnet20()


            latent_dim = model.fc.in_features

            if OutLayer=="CosLayer":
                from Models.Output_Layers.layer import CosineLayer
                # We replace the output layer by a cosine layer
                model.fc = CosineLayer(latent_dim, 10)
            elif OutLayer=="SLDA":
                from Models.Output_Layers.layer import SLDALayer
                # We replace the output layer by a cosine layer
                model.fc = SLDALayer(latent_dim, 10)
            elif OutLayer=="Linear_no_bias":
                model.fc = torch.nn.Linear(latent_dim, 10, bias=False)
            else:
                model.fc = torch.nn.Linear(latent_dim, 10, bias=True)

        else:
            model = Model(num_classes=scenario.nb_classes, OutLayer=OutLayer, pretrained_on=pretrained_on)
    return model.cuda()
