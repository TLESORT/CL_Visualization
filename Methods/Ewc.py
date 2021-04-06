import torch
from torch import nn
from torch.nn import functional as F
from copy import deepcopy
import os
import pickle
from torch.utils.data import DataLoader

from nngeometry.metrics import FIM
from nngeometry.layercollection import LayerCollection
from nngeometry.object import PMatDiag, PMatKFAC
from nngeometry.object.vector import PVector
import numpy as np

from Methods.trainer import Trainer


class EWC(Trainer):
    def __init__(self, config):
        super().__init__(config)
        self.layer_collection = LayerCollection.from_model(self.model)
        self.representation = config.representation

        self.importance = config.importance
        self.list_Fishers = {}

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


    def init_log(self, ind_task_log):
        super().init_log(ind_task_log)

        # temporar, we can not load FIM yet
        # and FIM computation will be skipped if first_task_loader is true
        if self.first_task_loaded and ind_task_log==0:
            self.list_Fishers[ind_task_log] = self.compute_fisher(ind_task_log, self.scenario_tr[ind_task_log], self.model)


    def init_task(self, ind_task, task_set):
        assert len(self.list_Fishers) == ind_task, print(f"{len(self.list_Fishers)} vs {ind_task}")
        return super().init_task(ind_task, task_set)

    def _can_load_first_task(self):
        #todo
        #path_fisher=os.path.join(self.log_dir, f"checkpoint_0_Fishers.pkl")

        #return os.path.isfile(path_fisher) and super()._can_load_first_task()
        return super()._can_load_first_task()

    def callback_task(self, ind_task, task_set):
        self.list_Fishers[ind_task] = self.compute_fisher(ind_task, task_set, self.model)
        super().callback_task(ind_task, task_set)

    def regularize_loss(self, model, loss):
        v = PVector.from_model(model)
        loss_regul = 0.

        assert not np.isnan(loss.item()), "Unfortunately, the loss is NaN (before regularization)"

        assert not np.isnan(v.norm().item()), "model weights"

        for i, (fim, v0) in self.list_Fishers.items():
            loss_regul += self.importance * fim.vTMv(v - v0)

            assert not np.isnan(loss_regul.item()), "Unfortunately, the loss is NaN"  # sanity check to detect nan

        return loss_regul+loss

    def load_log(self, ind_task=None):
        super().load_log(ind_task)
        #todo

        # file_name = os.path.join(self.log_dir, f"checkpoint_{ind_task}_Fishers.pkl")
        # with open(file_name, 'rb') as fp:
        #     list_Fishers_data = pickle.load(fp)
        #
        # for i, (fim_data, v0) in self.list_Fishers.items():
        #     fisher =
        #     list_Fishers_data[i] = [fim.data, v0]

    def post_training_log(self, ind_task=None):
        super().post_training_log(ind_task)
        #todo

        # list_Fishers_data = {}
        #
        # if (not self.fast) or ("ewc" in self.name_algo):
        #     # saving Fisher is slow but we need it for ewc
        #     file_name = os.path.join(self.log_dir, f"checkpoint_{ind_task}_Fishers.pkl")
        #     for i, (fim, v0) in self.list_Fishers.items():
        #         list_Fishers_data[i]=[fim.data, v0]
        #
        #     with open(file_name, 'wb') as f:
        #         pickle.dump(self.list_Fishers_data, f, pickle.HIGHEST_PROTOCOL)
        #
        # super().post_training_log(ind_task)

class EWC_id(EWC):
    def __init__(self, config):
        super().__init__(config)

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

class EWC_Diag(EWC):
    def __init__(self, config):
        super().__init__(config)

class EWC_Diag_id(EWC_id):
    def __init__(self, config):
        super().__init__(config)


class EWC_KFAC(EWC):
    def __init__(self, config):
        super().__init__(config)

class EWC_KFAC_id(EWC_id):
    def __init__(self, config):
        super().__init__(config)

