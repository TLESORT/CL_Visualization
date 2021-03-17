import torch
from torch import nn
from torch.nn import functional as F
from copy import deepcopy
from torch.utils.data import DataLoader
from torch.nn.utils.convert_parameters import _check_param_device, parameters_to_vector, vector_to_parameters

import torch.optim as optim
from collections import defaultdict

import torch
from types import MethodType

from methods.trainer import Trainer
from model import Model



def parameters_to_grad_vector(parameters):
    # Flag for the device where the parameter is located
    param_device = None
    vec = []
    for param in parameters:
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)
        vec.append(param.grad.view(-1))

    return torch.cat(vec)


def count_parameter(model):
    return sum(p.numel() for p in model.parameters())


class Storage(torch.utils.data.Dataset):
    """
    A dataset wrapper used as a memory to store the data
    """

    def __init__(self):
        super(Storage, self).__init__()
        self.storage = []

    def __len__(self):
        return len(self.storage)

    def __getitem__(self, index):
        return self.storage[index]

    def append(self, x):
        self.storage.append(x)

    def extend(self, x):
        self.storage.extend(x)


class Memory(Storage):
    def reduce(self, m):
        self.storage = self.storage[:m]


class OGD(Trainer):
    def __init__(self, args, root_dir, scenario_name, num_tasks, verbose, dev):
        super().__init__(args, root_dir, scenario_name, num_tasks, verbose, dev)
        self.algo_name = "ogd"

        # author comment: # Leave it to 0, this is for the case when using Lenet, projecting orthogonally only against the linear layers seems to work better
        self.all_features = 0

        self.memory_size = 100
        self.criterion = nn.CrossEntropyLoss()
        n_params = count_parameter(self.model)
        self.ogd_basis = torch.empty(n_params, 0).cuda()
        self.ogd_basis_ids = defaultdict(lambda: torch.LongTensor([]).cuda())

        self.task_memory = {}
        self.task_grad_memory = {}
        self.task_grad_mem_cache = {}

    def callback_task(self, ind_task, task_set):
        self.ogd_basis.cuda()
        self.update_mem(task_set, ind_task + 1)

    def update_mem(self, task_set, task_count):
        self.task_count = task_count

        num_sample_per_task = self.memory_size  # // (self.config.n_tasks-1)
        num_sample_per_task = min(len(task_set), num_sample_per_task)

        memory_length = []
        for i in range(task_count):
            memory_length.append(num_sample_per_task)

        for storage in self.task_memory.values():
            ## reduce the size of the stored elements
            storage.reduce(num_sample_per_task)

        self.task_memory[0] = Memory()  # Initialize the memory slot

        randind = torch.randperm(len(task_set))[:num_sample_per_task]  # randomly sample some data
        for ind in randind:  # save it to the memory

            self.task_memory[0].append(task_set[ind])

        ####################################### Grads MEM ###########################

        for storage in self.task_grad_memory.values():
            storage.reduce(num_sample_per_task)

        # if trainer.config.method in ['ogd', 'pca']:
        ogd_train_loader = torch.utils.data.DataLoader(self.task_memory[0],
                                                       batch_size=1,
                                                       shuffle=False,
                                                       num_workers=1)

        self.task_memory[0] = Memory()

        new_basis_tensor = self._get_new_ogd_basis(ogd_train_loader, self.opt).cpu()

        if self.test_label:
            if self.all_features:
                if hasattr(self.model, "conv"):
                    n_params = count_parameter(self.model.linear) + count_parameter(self.model.conv)
                else:
                    n_params = count_parameter(self.model.linear)
            else:

                n_params = count_parameter(self.model.linear)

        else:
            n_params = count_parameter(self.model)

        self.ogd_basis = torch.empty(n_params, 0).cpu()

        for t, mem in self.task_grad_memory.items():
            task_ogd_basis_tensor = torch.stack(mem.storage, axis=1).cpu()

            self.ogd_basis = torch.cat([self.ogd_basis, task_ogd_basis_tensor], axis=1).cpu()

        self.ogd_basis = self.orthonormalize(self.ogd_basis, new_basis_tensor, normalize=True)

        # (g) Store in the new basis
        ptr = 0

        for t in range(len(memory_length)):

            task_mem_size = memory_length[t]
            idxs_list = [i + ptr for i in range(task_mem_size)]

            self.ogd_basis_ids[t] = torch.LongTensor(idxs_list).cuda()

            self.task_grad_memory[t] = Memory()  # Initialize the memory slot

            length = task_mem_size
            for ind in range(length):  # save it to the memory
                self.task_grad_memory[t].append(self.ogd_basis[:, ptr].cpu())
                ptr += 1

    def _get_new_ogd_basis(self, train_loader, optimizer):
        new_basis = []

        for _, (x, y, t) in enumerate(train_loader):
            inputs = x.cuda()
            targets = y.cuda()
            task = t.cuda()

            if self.test_label:
                out = self.model.forward_task(inputs, t)
            else:
                out = self.model(inputs)

            assert out.shape[0] == 1

            pred = out[0, int(targets.item())].cpu()

            optimizer.zero_grad()
            pred.backward()

            ### retrieve  \nabla f(x) wrt theta
            new_basis.append(parameters_to_grad_vector(self.get_params_dict(last=False)).cpu())

        del out, inputs, targets
        torch.cuda.empty_cache()
        new_basis = torch.stack(new_basis).T

        return new_basis

    def project_vec(self, vec, proj_basis):
        if proj_basis.shape[1] > 0:  # param x basis_size
            dots = torch.matmul(vec, proj_basis)  # basis_size  dots= [  <vec, i >   for i in proj_basis ]
            out = torch.matmul(proj_basis, dots)  # out = [  <vec, i > i for i in proj_basis ]
            return out
        else:
            return torch.zeros_like(vec)

    def orthonormalize(self, main_vectors, additional_vectors, normalize=True):
        ## orthnormalize the basis (graham schmidt)
        for element in range(additional_vectors.size()[1]):
            coeff = torch.mv(main_vectors.t(), additional_vectors[:, element])  ## x - <x,y>y/ ||<x,y>||
            pv = torch.mv(main_vectors, coeff)
            d = (additional_vectors[:, element] - pv) / torch.norm(additional_vectors[:, element] - pv, p=2)
            main_vectors = torch.cat((main_vectors, d.view(-1, 1)), dim=1)
            del pv
            del d
        return main_vectors.cuda()

    def optimizer_step(self, ind_task):

        task_key = str(ind_task)

        ### take gradients with respect to the parameters
        grad_vec = parameters_to_grad_vector(self.get_params_dict(last=False))
        cur_param = parameters_to_vector(self.get_params_dict(last=False))

        # if self.config.method in ['ogd', 'pca']:
        proj_grad_vec = self.project_vec(grad_vec,
                                         proj_basis=self.ogd_basis)
        ## take the orthogonal projection
        new_grad_vec = grad_vec - proj_grad_vec

        ### SGD update  =>  new_theta= old_theta - learning_rate x ( derivative of loss function wrt parameters )
        cur_param -= self.lr * new_grad_vec  # .to(self.config.device)

        vector_to_parameters(cur_param, self.get_params_dict(last=False))

        # if test_label ==> multi-head
        if self.test_label:
            # Update the parameters of the last layer without projection, when there are multiple heads)
            cur_param = parameters_to_vector(self.get_params_dict(last=True, ind_task=ind_task))
            grad_vec = parameters_to_grad_vector(self.get_params_dict(last=True, ind_task=ind_task))
            cur_param -= self.lr * grad_vec
            vector_to_parameters(cur_param, self.get_params_dict(last=True, ind_task=ind_task))

        ### zero grad
        self.opt.zero_grad()

    def get_params_dict(self, last, ind_task=None):
        # if ind_task is not None:
        #     task_key = str(ind_task)

        if self.test_label:
            if last:
                return self.model.list_heads[ind_task].parameters()
            else:

                if self.all_features:
                    ## take the conv parameters into account
                    if hasattr(self.model, "conv"):
                        return list(self.model.linear.parameters()) + list(self.model.conv.parameters())
                    else:
                        return self.model.linear.parameters()


                else:
                    return self.model.linear.parameters()
                # return self.model.linear.parameters()
        else:
            return self.model.parameters()
