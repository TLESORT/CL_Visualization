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
import numpy as np


class EWC(Trainer):
    def __init__(self, args, root_dir, scenario_name, num_tasks, representation, verbose, dev):
        super().__init__(args, root_dir, scenario_name, num_tasks, verbose, dev)
        self.layer_collection = LayerCollection.from_model(self.model)
        self.representation = representation

        self.importance = args.importance
        self.list_Fishers = {}
        self.representation = representation

    def compute_fisher(self, ind_task, fisher_set, model):
        fisher_loader = DataLoader(fisher_set, batch_size=264, shuffle=True, num_workers=6)

        fim = FIM(model=model,
                     loader=fisher_loader,
                     representation=self.representation,
                     n_output=self.scenario_tr.nb_classes,
                     variant='classif_logits',
                     device='cuda')

        v0 = PVector.from_model(model).clone().detach()

        assert not np.isnan(v0.norm().item())

        assert not np.isnan(fim.frobenius_norm().item()), "There are Nan in the Fisher Matrix"

        return fim, v0

    def callback_task(self, ind_task, task_set):
        self.list_Fishers[ind_task] = self.compute_fisher(ind_task, task_set, self.model)

    def regularize_loss(self, model, loss):
        v = PVector.from_model(model)
        loss_regul = 0.

        assert not np.isnan(loss.item()), "Unfortunately, the loss is NaN (before regularization)"

        assert not np.isnan(v.norm().item()), "model weights"

        for i, (fim, v0) in self.list_Fishers.items():
            loss_regul += self.importance * fim.vTMv(v - v0)

            assert not np.isnan(loss_regul.item()), "Unfortunately, the loss is NaN"  # sanity check to detect nan

        return loss_regul+loss


class EWC_Diag(EWC):
    def __init__(self, args, root_dir, scenario_name, num_tasks, verbose, dev):
        super().__init__(args, root_dir, scenario_name, num_tasks, PMatDiag, verbose, dev)
        self.algo_name = "ewc_diag"

class EWC_Diag_id(EWC_Diag):
    def __init__(self, args, root_dir, scenario_name, num_tasks, verbose, dev):
        super().__init__(args, root_dir, scenario_name, num_tasks, verbose, dev)
        self.algo_name = "ewc_diag_id"
        #self.small_lambda = 0.001

    def compute_fisher(self, ind_task, fisher_set, model):
        fim, v0 = super().compute_fisher(ind_task, fisher_set, model)
        #small_lambda = 0.0001 * fim.trace().max().item()
        small_lambda = 0.01

        return fim, v0, small_lambda

    def regularize_loss(self, model, loss):
        v = PVector.from_model(model)
        weight_decay = 0.0

        assert not np.isnan(loss.item()), "Unfortunately, the loss is NaN (before regularization)"

        assert not np.isnan(v.norm().item()), "model weights"

        for i, (fim, v0, small_lambda) in self.list_Fishers.items():
            if (v - v0).norm() > 0:
                weight_decay += small_lambda * (v - v0).norm()**2
            loss += self.importance * fim.vTMv(v - v0)

            assert not np.isnan(loss.item()), "Unfortunately, the loss is NaN"  # sanity check to detect nan
        if weight_decay > 0:
            loss = loss + weight_decay
        return loss

class EWC_KFAC_id(EWC_Diag_id):
    def __init__(self, args, root_dir, scenario_name, num_tasks, verbose, dev):
        super().__init__(args, root_dir, scenario_name, num_tasks, verbose, dev)

class EWC_KFAC(EWC):
    def __init__(self, args, root_dir, scenario_name, num_tasks, verbose, dev):
        super().__init__(args, root_dir, scenario_name, num_tasks, PMatKFAC, verbose, dev)
        self.algo_name = "ewc_kfac"
