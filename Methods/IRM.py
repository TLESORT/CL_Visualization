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
    return nn.functional.binary_cross_entropy_with_logits(logits, y.float())


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

    def get_env_data(self, env_id, ind_task, train=True):
        ''''Here an env is the data in the memory, except for the env for current task'''
        if train:
            assert env_id < ind_task

        # get data corresponding to env_id <=> ind_task
        if env_id < ind_task:
            indexes = self.data_memory.get_indexes_task(env_id)
            x, y, _ = self.data_memory.get_samples(indexes)
        else:
            AssertionError("get env function is only made for past data")

        return x, y


class IB_IRM(Base_IRM):
    def __init__(self, config):
        super().__init__(config)

        self.p_weight = 1e2  # penalty_weight: [0, 1e2, 1e3, 1e4, 1e5, 1e6]
        self.ib_lambda = 1e-1  # [0, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
        self.class_condition = False  # define which type of Info bottleneck is use
        self.l2_weight = 0.001  # l2_regularizer_weight
        self.penalty_anneal_iters = 100
        self.ib_step = 0

        assert self.masked_out is None

    def init_task(self, ind_task, task_set):
        self.step = 0
        return super().init_task(ind_task, task_set)

    def head_with_grad(self, x_, y_, t_, ind_task, epoch):

        if ind_task == 0:
            return super().head_with_grad(x_, y_, t_, ind_task, epoch)
        else:
            self.opt.zero_grad()
            if self.test_label:
                output = self.model.forward_task(x_, t_)
            else:
                output = self.model(x_)

            loss = self.loss_ib_irm(x_, y_, ind_task)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # clip gradient to avoid Nan
            loss = self.regularize_loss(self.model, loss)
            self.optimizer_step(ind_task)

        return output, loss

    def loss_ib_irm(self, current_x, current_y, ind_task):

        step = self.step

        list_nll = []
        list_penalty = []
        list_var = []

        # for id_env, env in enumerate(train_envs):
        for id_env in range(ind_task+1):
            # get one batch of the env
            if id_env == ind_task:
                batch_x, batch_y = current_x, current_y
            else:
                batch_x, batch_y = self.get_env_data(id_env, ind_task)

            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            # the model should return the logits and the latent vector of a specific layer
            #logits, inter_logits = model(batch_x, layer_latent_vector=-1)

            inter_logits = self.model.feature_extractor(batch_x) # equivalent to layer_latent_vector=-1
            logits = self.model.head(inter_logits)

            # DEV ##
            # binarize output
            if logits.shape[1] == 2:
                # one value
                logits = (logits[:, 1] - logits[:, 0])
            else:
                AssertionError("Code not compatible with more than two classes")
            ########

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
        for w in self.model.parameters():
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

        self.step += 1
        return loss
