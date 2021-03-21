from continuum.datasets import MNIST
from continuum.datasets import CIFAR10
from continuum.datasets import MNISTFellowship


def get_dataset(path_dir, name_dataset, name_scenario, train="True"):
    if name_dataset == "MNIST":
        dataset = MNIST(path_dir, download=True, train=train)
    if name_dataset == "CIFAR10":
        dataset = CIFAR10(path_dir, download=True, train=train)
    elif name_dataset == "mnist_fellowship" and name_scenario=="Disjoint":
        dataset = MNISTFellowship(path_dir, download=True, train=train)
    elif name_dataset == "mnist_fellowship" and name_scenario=="Domain":
        dataset = MNISTFellowship(path_dir, download=True, train=train, update_labels=False)
    else:
        print("Dataset unKnown")
    return dataset
