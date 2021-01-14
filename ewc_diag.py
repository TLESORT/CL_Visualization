import torch
from torch import nn
from torch.nn import functional as F
from copy import deepcopy
from torch.utils import data

from nngeometry.object import PVector, PMatDiag
from nngeometry.metrics import FIM_MonteCarlo
from nngeometry.layercollection import LayerCollection

from trainer import Trainer


class EWC_Diag(Trainer):
    def __init__(self, scenario, continuum, model):
        super().__init__(scenario, continuum, model)
        self.model = model
        self.layer_collection = LayerCollection.from_model(model)
        self.importance = 100
        self.list_Fishers = {}
        self.algo_name = "ewc_diag"

    def compute_fisher(self, model, ind_task):


        self.continuum.set_task(ind_task)
        F_diag = FIM_MonteCarlo(model=model,
                                 loader=self.train_loader,
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
