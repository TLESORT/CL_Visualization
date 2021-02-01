import torch
from torch import nn
from torch.nn import functional as F
from copy import deepcopy
from torch.utils import data

from nngeometry.metrics import FIM
from nngeometry.layercollection import LayerCollection
from nngeometry.object import PMatDiag, PMatKFAC
from nngeometry.object.vector import PVector
from trainer import Trainer

class EWC_Diag(Trainer):
    def __init__(self, root_dir, dataset, scenario_name, model, num_tasks):
        super().__init__(root_dir, dataset, scenario_name, model, num_tasks, PMatDiag)


class EWC_KFAC(Trainer):
    def __init__(self, root_dir, dataset, scenario_name, model, num_tasks):
        super().__init__(root_dir, dataset, scenario_name, model, num_tasks, PMatKFAC)


class EWC(Trainer):
    def __init__(self, root_dir, dataset, scenario_name, model, num_tasks, representation):
        super().__init__(root_dir, dataset, scenario_name, model, num_tasks)
        self.model = model
        self.layer_collection = LayerCollection.from_model(model)

        self.importance = 100
        self.list_Fishers = {}
        self.algo_name = "ewc_diag"
        self.representation=representation

    def compute_fisher(self, task_set, model, ind_task):
        fisher_set = deepcopy(task_set)

        F_diag = FIM(model=model,
                     loader=fisher_set,
                     representation=PMatDiag,
                     device='cuda')

        v0 = PVector.from_model(model).clone().detach()

        return F_diag, v0


def init_task(self, ind_task):
    pass


def callback_task(self, ind_task):
    self.list_Fishers[ind_task] = self.compute_fisher(self.model, ind_task)


def regularize_loss(self, model, loss):
    v = PVector.from_model(model)

    for i, (fim, v0) in self.list_Fishers.items():
        loss += self.importance * fim.vTMv(v - v0)

    assert loss == loss  # sanity check to detect nan

    return loss


class EWC_Diag(Trainer):
    def __init__(self, root_dir, dataset, scenario_name, model, num_tasks):
        super().__init__(root_dir, dataset, scenario_name, model, num_tasks)
        self.model = model
        self.layer_collection = LayerCollection.from_model(model)

        self.importance = 100
        self.list_Fishers = {}
        self.algo_name = "ewc_diag"
        self.representation=PMatDiag

    def compute_fisher(self, task_set, model, ind_task):
        fisher_set = deepcopy(task_set)

        F_diag = FIM(model=model,
                     loader=fisher_set,
                     representation=PMatDiag,
                     device='cuda')

        v0 = PVector.from_model(model).clone().detach()

        return F_diag, v0


def init_task(self, ind_task):
    pass


def callback_task(self, ind_task):
    self.list_Fishers[ind_task] = self.compute_fisher(self.model, ind_task)


def regularize_loss(self, model, loss):
    v = PVector.from_model(model)

    for i, (fim, v0) in self.list_Fishers.items():
        loss += self.importance * fim.vTMv(v - v0)

    assert loss == loss  # sanity check to detect nan

    return loss
