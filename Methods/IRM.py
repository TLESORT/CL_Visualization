import copy

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import pdb

from Methods.rehearsal import Rehearsal

#from domainbed import networks
#from domainbed.lib.misc import random_pairs_of_minibatches

ALGORITHMS = [
    'ERM',
    'xylERM',
    'VREx',
    'IRM',
    'GroupDRO',
    'Mixup',
    'MLDG',
    'CORAL',
    'MMD',
    'DANN',
    'CDANN',
    'MTL',
    'SagNet',
    'ARM',
]


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():  # global() is a dict
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class OOD_Algorithm(Rehearsal):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """

    def __init__(self, config):
        super().__init__(config)
        # def __init__(self, input_shape, num_classes, num_domains, hparams):
        #     super(Algorithm, self).__init__()
        self.config = config
        self.hparams = config
        self.step = 0

    def update(self, minibatches):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

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

        return x.cuda(), y.cuda()


class ERM(OOD_Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, config):
        super().__init__(config)

        self.featurizer = self.model.feature_extractor
        self.classifier = self.model.head
        self.network = self.model

        self.xyl = False
        self.xylopt = False
        if self.xylopt:
            print("LRRRRRRR:", self.config.lr)
            self.xyl = True
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.lr,
                momentum=0.9, weight_decay=self.config.weight_decay)

            print("DECAY STEP:", config.sch_size)
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=config.sch_size, gamma=0.1)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr,
                weight_decay=config.weight_decay)

    def head_with_grad(self, x_, y_, t_, ind_task, epoch):

        if ind_task == 0:
            return super().head_with_grad(x_, y_, t_, ind_task, epoch)
        else:
            self.opt.zero_grad()
            if self.test_label:
                output = self.model.forward_task(x_, t_)
            else:
                output = self.model(x_)

            loss = self.loss_ood(x_, y_, ind_task)
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # clip gradient to avoid Nan
            #loss = self.regularize_loss(self.model, loss)
            self.optimizer_step(ind_task)
        if self.xyl:
            self.scheduler.step()

        return output, loss

    def loss_ood(self, current_x, current_y, ind_task):

        step = self.step
        loss = 0.0
        # for id_env, env in enumerate(train_envs):
        for id_env in range(ind_task+1):
            # get one batch of the env
            if id_env == ind_task:
                batch_x, batch_y = current_x, current_y
            else:
                batch_x, batch_y = self.get_env_data(id_env, ind_task)

            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

            loss += F.cross_entropy(self.predict(batch_x), batch_y)

        return loss

    def predict(self, x):
        return self.model(x)


class IBERM(ERM):

    def __init__(self, config):
        super(IBERM, self).__init__(config)
        self.model.register_buffer('update_count', torch.tensor([0]))
        self.normalize = config.normalize
        self.ib_penalty_anneal_iters = 0

    def loss_ood(self, current_x, current_y, ind_task):

        #def update(self, minibatches):
        minibatches = []
        for id_env in range(ind_task):
            minibatches.append(self.get_env_data(id_env, ind_task))
        minibatches.append([current_x, current_y])

        penalty_weight = (self.config.ib_lambda if self.model.update_count
                       >= self.ib_penalty_anneal_iters else 0.0)
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])

        logits = self.featurizer(all_x)
        loss = F.cross_entropy(self.classifier(logits), all_y)

        var_loss = logits.var(dim=0).mean()

        loss += penalty_weight * var_loss
        if self.normalize and penalty_weight > 1:
            loss /= (1 + penalty_weight)

        if self.model.update_count == self.ib_penalty_anneal_iters:
            if not self.xyl:
                print("!!!!UPDATE IB-ERM ADAM OPTIMIZER")
                self.optimizer = torch.optim.Adam(
                    self.network.parameters(),
                    lr=self.config.lr,
                    weight_decay=self.config.weight_decay)
        return loss

class IRM(ERM):
    """Invariant Risk Minimization"""

    def __init__(self, config):
        super().__init__(config)
        self.model.register_buffer('update_count', torch.tensor([0]))
        self.normalize = config.normalize
        self.irm_penalty_anneal_iters = 0

    @staticmethod
    def _irm_penalty(logits, y):
        scale = torch.tensor(1.).to(y.device).requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result


    def loss_ood(self, current_x, current_y, ind_task):

        #def update(self, minibatches):
        minibatches = []
        for id_env in range(ind_task):
            minibatches.append(self.get_env_data(id_env, ind_task))
        minibatches.append([current_x, current_y])

        envs_batches = minibatches

        penalty_weight = (self.config.irm_lambda if self.model.update_count
                    >= self.irm_penalty_anneal_iters else 0.0)  # todo

        nll = 0.
        penalty = 0.
        all_x = torch.cat([x for x, y in envs_batches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            penalty += self._irm_penalty(logits, y)
        nll /= len(minibatches)
        penalty /= len(minibatches)
        loss = nll + (penalty_weight * penalty)
        if self.normalize and penalty_weight > 1:
            loss /= (1 + penalty_weight)

        if self.model.update_count == self.irm_penalty_anneal_iters:
            if not self.xyl:
                print("!!!!UPDATE IRM ADAM OPTIMIZER")
                self.optimizer = torch.optim.Adam(
                    self.network.parameters(),
                    lr=self.config.lr,
                    weight_decay=self.config.weight_decay)

        self.model.update_count += 1

        return loss

class IBIRM(IRM):
    """Invariant Risk Minimization"""

    def __init__(self, config):
        super(IBIRM, self).__init__(config)

        self.irm_penalty_anneal_iters = 0
        self.ib_penalty_anneal_iters = 0

    def loss_ood(self, current_x, current_y, ind_task):

        # def update(self, minibatches):
        minibatches = []
        for id_env in range(ind_task):
            minibatches.append(self.get_env_data(id_env, ind_task))
        minibatches.append([current_x, current_y])

        penalty_weight = (self.config.irm_lambda if self.model.update_count
                    >= self.irm_penalty_anneal_iters else 0.0)  # todo

        ib_penalty_weight = (self.config.ib_lambda if self.model.update_count
                   >= self.ib_penalty_anneal_iters else 0.0)

        nll = 0.
        penalty = 0.
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        inter_logits = self.featurizer(all_x)
        all_logits = self.classifier(inter_logits)
        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            penalty += self._irm_penalty(logits, y)
        nll /= len(minibatches)
        penalty /= len(minibatches)
        loss = nll + (penalty_weight * penalty)
        if self.normalize and penalty_weight > 1:
            loss /= (1 + penalty_weight)

        var_loss = inter_logits.var(dim=0).mean()
        loss += ib_penalty_weight * var_loss

        if self.model.update_count == self.config.irm_penalty_anneal_iters:
            if not self.xyl:
                print("!!!!UPDATE IB-ERM ADAM OPTIMIZER")
                self.optimizer = torch.optim.Adam(
                    self.network.parameters(),
                    lr=self.config.lr,
                    weight_decay=self.config.weight_decay)

        self.model.update_count += 1
        return loss