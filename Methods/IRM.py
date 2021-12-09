import copy

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import pdb

from Methods.rehearsal import Rehearsal

# from domainbed import networks
# from domainbed.lib.misc import random_pairs_of_minibatches

# TODO test in terra INcognita


ALGORITHMS = [
    'ERM',  # TEST
    'Fish',
    'IRM', # TEST
    'GroupDRO',  # TODO (eventually)
    'Mixup',
    'MLDG',
    'CORAL',
    'MMD',
    'DANN', # TODO (eventually)
    'CDANN',
    'MTL',
    'SagNet',
    'ARM',
    'VREx',
    'RSC',
    'SD',
    'ANDMask',
    'SANDMask',
    'IGA',
    'SelfReg',
    "Fishr",
    'TRM',
    'IB_ERM', # TEST
    'IB_IRM', # TEST
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
    Empirical Risk Minimization (ERM)  (Somehow Same as Rehearsal Base)
    """

    def __init__(self, config):
        super().__init__(config)

        # self.featurizer = self.model.feature_extractor
        # self.classifier = self.model.head
        # self.network = self.model


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
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # clip gradient to avoid Nan
            # loss = self.regularize_loss(self.model, loss)
            self.optimizer_step(ind_task)

        return output, loss

    def get_minibatches(self, current_x, current_y, ind_task):
        minibatches = []
        for id_env in range(ind_task):
            minibatches.append(self.get_env_data(id_env, ind_task))
        minibatches.append([current_x, current_y])
        return minibatches

    def loss_ood(self, current_x, current_y, ind_task):
        minibatches = self.get_minibatches(current_x, current_y, ind_task)
        #all_x = torch.cat([x for x, y in minibatches])
        all_hat_y = torch.cat([self.predict(x) for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])

        loss = F.cross_entropy(all_hat_y, all_y)

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

        ib_penalty_weight = (self.ib_lambda if self.update_count
                                                          >= self.ib_penalty_anneal_iters else
                             0.0)
        minibatches = self.get_minibatches(current_x, current_y, ind_task)

        nll = 0.
        ib_penalty = 0.

        #all_x = torch.cat([x for x, y in minibatches])
        all_features = torch.cat([self.featurizer(x) for x, y in minibatches])
        #all_features = self.featurizer(all_x)
        all_logits = self.classifier(all_features)
        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            features = all_features[all_logits_idx:all_logits_idx + x.shape[0]]
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            ib_penalty += features.var(dim=0).mean()

        nll /= len(minibatches)
        ib_penalty /= len(minibatches)

        # Compile loss
        loss = nll
        loss += ib_penalty_weight * ib_penalty

        if self.model.update_count == self.ib_penalty_anneal_iters:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                list(self.model.parameters()),
                lr=self.lr,
                weight_decay=self.weight_decay)
        self.model.update_count += 1
        return loss


class IRM(ERM):
    """Invariant Risk Minimization"""

    def __init__(self, config):
        super().__init__(config)
        self.model.register_buffer('update_count', torch.tensor([0]))
        self.normalize = config.normalize
        self.irm_penalty_anneal_iters = 500  # ('irm_penalty_anneal_iters', 500, int(10 ** random_state.uniform(0, 4)))

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

        minibatches = self.get_minibatches(current_x, current_y, ind_task)

        penalty_weight = (self.config.irm_lambda if self.model.update_count
                                                    >= self.irm_penalty_anneal_iters else 0.0)  # todo

        nll = 0.
        penalty = 0.
        # all_x = torch.cat([x for x, y in minibatches])
        # all_logits = self.model(all_x)
        all_logits = torch.cat([self.model(x) for x, y in minibatches])
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
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay)

        self.model.update_count += 1

        return loss


class IBIRM(IRM):
    """Invariant Risk Minimization"""

    def __init__(self, config):
        super(IBIRM, self).__init__(config)

        self.irm_penalty_anneal_iters = 500  # ('irm_penalty_anneal_iters', 500, int(10 ** random_state.uniform(0, 4)))
        self.ib_penalty_anneal_iters = 500  # ('ib_penalty_anneal_iters', 500, int(10 ** random_state.uniform(0, 4)))

    def loss_ood(self, current_x, current_y, ind_task):

        minibatches = self.get_minibatches(current_x, current_y, ind_task)

        irm_penalty_weight = (self.config.irm_lambda if self.model.update_count
                                                    >= self.irm_penalty_anneal_iters else 1.0)

        ib_penalty_weight = (self.config.ib_lambda if self.model.update_count
                                                      >= self.ib_penalty_anneal_iters else 0.0)

        nll = 0.
        irm_penalty = 0.
        ib_penalty = 0.

        #all_x = torch.cat([x for x, y in minibatches])
        all_features = torch.cat([self.model.feature_extractor(x) for x, y in minibatches])
        #all_features = self.model.feature_extractor(all_x)
        all_logits = self.model.head(all_features)
        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            features = all_features[all_logits_idx:all_logits_idx + x.shape[0]]
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            irm_penalty += self._irm_penalty(logits, y)
            ib_penalty += features.var(dim=0).mean()

        nll /= len(minibatches)
        irm_penalty /= len(minibatches)
        ib_penalty /= len(minibatches)

        # Compile loss
        loss = nll
        loss += irm_penalty_weight * irm_penalty
        loss += ib_penalty_weight * ib_penalty

        if self.model.update_count == self.irm_penalty_anneal_iters or self.model.update_count == self.ib_penalty_anneal_iters:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                list(self.model.parameters()),
                lr=self.lr,
                weight_decay=self.weight_decay)

        self.model.update_count += 1
        return loss


class SpectralDecoupling(ERM):
    """SpectralDecoupling: FROM GRADIENT STARVATION PAPER."""

    def __init__(self, config):
        super().__init__(config)
        self.sp_coef = 0.003



    def loss_ood(self, current_x, current_y, ind_task):

        minibatches = self.get_minibatches(current_x, current_y, ind_task)

        #all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        #y_hat = self.classifier(self.featurizer(all_x))
        all_hat_y = torch.cat([self.predict(x) for x, y in minibatches])
        loss = torch.log(1.0 + torch.exp(-all_hat_y[:, 0] * all_y)).mean()
        loss += self.sp_coef * (all_hat_y ** 2).mean()

        return loss

    def optimizer_step(self, ind_task):
        self.opt.step()
        if self.num_classes == 2:
            with torch.no_grad():
                # binary classification vectors are just opposit to each others
                self.classifier.layer.weight[1,:] = -1 * self.classifier.layer.weight[0,:]


