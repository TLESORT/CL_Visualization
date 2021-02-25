from continuum.datasets import MNIST
from continuum.datasets import MNISTFellowship


def get_dataset(path_dir, name_dataset, train="True"):
    if name_dataset == "MNIST":
        dataset = MNIST(path_dir, download=True, train=train)
    elif name_dataset == "mnist_fellowship_merge":
        dataset = MNISTFellowship(path_dir, download=True, train=train)
    elif name_dataset == "mnist_fellowship":
        dataset = MNISTFellowship(path_dir, download=True, train=train)
    else:
        print("Dataset unKnown")
    return dataset
