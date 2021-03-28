
import torch
from torchvision.datasets import SVHN

from continuum.datasets import PyTorchDataset
from continuum.datasets import MNIST
from continuum.datasets import CIFAR10
from continuum.datasets import MNISTFellowship


from Models.model import Model
from Models.multihead_model import MultiHead_Model
from Models.resnet import cifar_resnet20
from Models.layer import CosineLayer

def get_dataset(path_dir, name_dataset, name_scenario, train="True"):
    if name_dataset == "MNIST":
        dataset = MNIST(path_dir, download=True, train=train)
    elif name_dataset == "CIFAR10":
        dataset = CIFAR10(path_dir, download=True, train=train)
    elif name_dataset == "SVHN":
        dataset = PyTorchDataset("path_dir", dataset_type=SVHN, train=True, download=True)
    elif name_dataset == "mnist_fellowship" and name_scenario=="Disjoint":
        dataset = MNISTFellowship(path_dir, download=True, train=train)
    elif name_dataset == "mnist_fellowship" and name_scenario=="Domain":
        dataset = MNISTFellowship(path_dir, download=True, train=train, update_labels=False)
    else:
        print("Dataset unKnown")
    return dataset



def get_model(name_dataset, scenario, pretrained, test_label, cosLayer, method):
    if test_label:
        # there are no test label for domain incremental since the classes should be always the same
        #assert name_dataset == "Disjoint"

        list_classes_per_tasks = []
        for task_set in scenario:
            classes = task_set.get_classes()
            list_classes_per_tasks.append(classes)

        model = MultiHead_Model(num_classes=scenario.nb_classes,
                                classes_per_tasks=list_classes_per_tasks,
                                cosLayer=cosLayer,
                                method=method).cuda()
    else:

        if name_dataset == "CIFAR10" or name_dataset == "CIFAR100" or name_dataset == "SVHN":

            if pretrained:
                model = cifar_resnet20(pretrained="CIFAR100")
                model.num_classes = 10 # manual correction
                for param in model.parameters():
                    param.requires_grad = False
            else:
                model = cifar_resnet20()


            latent_dim = model.fc.in_features

            if cosLayer:
                # We replace the output layer by a cosine layer
                model.fc = CosineLayer(latent_dim, 10)
            else:
                model.fc = torch.nn.Linear(latent_dim, 10, bias=False)

        else:
            model = Model(num_classes=scenario.nb_classes, cosLayer=cosLayer)
    return model
