import torch
from torch import nn, autograd
from torch.nn import functional as F
from copy import deepcopy
import os
import pickle
from torch.utils.data import DataLoader

import numpy as np

from Methods.rehearsal import Rehearsal


def mean_nll(logits, y):
    return nn.functional.binary_cross_entropy_with_logits(logits, y)


def mean_accuracy(logits, y):
    preds = (logits > 0.).float()
    return ((preds - y).abs() < 1e-2).float().mean()


def penalty(logits, y):
    scale = torch.tensor(1.).to(y.device).requires_grad_()
    loss = mean_nll(logits * scale, y)
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad ** 2)

class Base_IRM(Rehearsal):

    def __init__(self, config):
        super().__init__(config)

    def get_env_data(self, env_id, train=True):
        ''''Here an env is the data in the memory, except for the env for current task'''
        if train:
            assert env_id <= self.ind_task

        # get data corresponding to env_id <=> ind_task
        if env_id < self.ind_task:
            indexes = self.data_memory.get_indexes_task(env_id)
            x, y, _ = self.get_samples(indexes)
        else:
            # TODO
            # get the current data
            pass

        return x, y


class IB_IRM(Base_IRM):
    def __init__(self, config):
        super().__init__(config)

        self.p_weight = 1e2  # penalty_weight: [0, 1e2, 1e3, 1e4, 1e5, 1e6]
        self.ib_lambda = 1e-1  # [0, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
        self.class_condition = False  # what does this variable means?
        self.l2_weight = 0.001  # l2_regularizer_weight
        self.penalty_anneal_iters = 100
        self.ib_step = 0

    def loss_ib_irm(self, model, train_envs, epoch, ind_task):

        step = epoch  # I hypothetize that this is the same

        list_nll = []
        list_penalty = []
        list_var = []

        #for id_env, env in enumerate(train_envs):
        for id_env in enumerate(ind_task):
            # get one batch of the env
            batch_x, batch_y = self.get_env_data(id_env)
            # the model should return the logits and the latent vector of a specific layer
            logits, inter_logits = model(batch_x, layer_latent_vector=-1)
            list_nll.append(mean_nll(logits, batch_y))
            list_penalty.append(penalty(logits, batch_y))

            if self.ib_lambda > 0.:
                if self.class_condition:
                    num_classes = 2
                    index = [batch_y.squeeze() == i for i in range(num_classes)]
                    var = sum(inter_logits[ind].var(dim=0).mean() for ind in index)
                    var /= num_classes
                else:
                    var = inter_logits.var(dim=0).mean()
                list_var.append(var)

        train_nll = torch.stack(list_nll).mean()

        weight_norm = torch.tensor(0.).to(0)
        for w in model.parameters():
            weight_norm += w.norm().pow(2)

        loss = train_nll.clone()
        loss += self.l2_weight * weight_norm

        # "info bottleneck"
        if self.ib_lambda > 0.:
            ib_weight = self.ib_lambda if step >= self.ib_step else 0.  # (?)
            var_loss = torch.stack(list_var).mean()
            loss += ib_weight * var_loss

        train_penalty = torch.stack(list_penalty).mean()
        penalty_weight = (self.p_weight if step >= self.penalty_anneal_iters else .0)
        loss += penalty_weight * train_penalty
        if penalty_weight > 1.0:
            loss /= penalty_weight

        return loss


