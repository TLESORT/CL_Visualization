
import numpy as np

def sampling(dataset, nb_samples, model, method="rand"):

    if method == "rand":
        inds = RandSampling(dataset, nb_samples)
    elif method == "grad":
        inds = GradSampling(dataset, model, nb_samples)
    else:
        raise NotImplementedError(f"Not implemented sampling strategy: {method}")

    return inds

def RandSampling(dataset, nb_samples):
    """"
    Sample randomly a set of data a return a subsample of the data
    """
    indexes = np.random.randint(0, len(dataset._y), nb_samples)
    return indexes


def GradSampling(dataset, model, nb_samples):
    """"
    Sample randomly a set of data a return a subsample of the data
    """
    #TODO
