import torch
from torch import nn
from torch.nn import functional as F
from copy import deepcopy
from torch.utils.data import DataLoader

from nngeometry.metrics import FIM
from nngeometry.layercollection import LayerCollection
from nngeometry.object import PMatDiag, PMatKFAC
from nngeometry.object.vector import PVector
from trainer import Trainer


class EWC(Trainer):
    def __init__(self, root_dir, dataset, scenario_name, model, num_tasks, representation, verbose, dev):
        super().__init__(root_dir, dataset, scenario_name, model, num_tasks, verbose, dev)
        self.model = model
        self.layer_collection = LayerCollection.from_model(model)
        self.representation = representation

        self.importance = 100
        self.list_Fishers = {}
        self.representation = representation

    def compute_fisher(self, ind_task, task_set, model):
        fisher_set = deepcopy(task_set)

        F_diag = FIM(model=model,
                     loader=DataLoader(fisher_set),
                     representation=self.representation,
                     n_output=self.scenario_tr.nb_classes,
                     variant='classif_logits',
                     device='cuda')

        v0 = PVector.from_model(model).clone().detach()

        return F_diag, v0

    def init_task(self, ind_task, task_set):
        return task_set

    def callback_task(self, ind_task, task_set):
        self.list_Fishers[ind_task] = self.compute_fisher(ind_task, task_set, self.model)

    def regularize_loss(self, model, loss):
        v = PVector.from_model(model)

        for i, (fim, v0) in self.list_Fishers.items():
            loss += self.importance * fim.vTMv(v - v0)

        assert loss == loss  # sanity check to detect nan

        return loss


class EWC_Diag(EWC):
    def __init__(self, root_dir, dataset, scenario_name, model, num_tasks, verbose, dev):
        super().__init__(root_dir, dataset, scenario_name, model, num_tasks, PMatDiag, verbose, dev)
        self.algo_name = "ewc_diag"


class EWC_KFAC(EWC):
    def __init__(self, root_dir, dataset, scenario_name, model, num_tasks, verbose, dev):
        super().__init__(root_dir, dataset, scenario_name, model, num_tasks, PMatKFAC, verbose, dev)
        self.algo_name = "ewc_kfac"
