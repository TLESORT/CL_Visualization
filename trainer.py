import torch
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

from torch.utils.data import DataLoader

import torch.optim as optim
import torch.nn.functional as F
import pickle
import os
import numpy as np
from copy import deepcopy

from utils import get_dataset
from model import Model
from continuum import ClassIncremental, InstanceIncremental
from continuum import Rotations

from eval import Continual_Evaluation


class Trainer(Continual_Evaluation):
    def __init__(self, args, root_dir, scenario_name, num_tasks, verbose, dev):

        super().__init__(args)

        self.lr = args.lr
        self.root_dir = root_dir
        self.verbose = verbose
        self.batch_size = args.batch_size

        self.dir_data = os.path.join(self.root_dir, "../../Datasets/")
        if not os.path.exists(self.dir_data):
            os.makedirs(self.dir_data)
        self.log_dir = os.path.join(self.root_dir, "Logs", scenario_name)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.sample_dir = os.path.join(self.root_dir, "Samples")
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)

        self.algo_name = "baseline"
        self.scenario_name = scenario_name
        self.dev = dev
        self.nb_epochs = args.nb_epochs

        dataset_train = get_dataset(self.dir_data, args.dataset, self.scenario_name, train=True)
        dataset_test = get_dataset(self.dir_data, args.dataset, self.scenario_name, train=False)

        scenario = None
        if self.scenario_name == "Rotations":
            self.scenario_tr = Rotations(dataset_train, nb_tasks=num_tasks)
            # NO SATISFYING SOLUTION YET HERE
        elif self.scenario_name == "Disjoint":
            self.scenario_tr = ClassIncremental(dataset_train, nb_tasks=num_tasks)
            self.scenario_te = ClassIncremental(dataset_test, nb_tasks=num_tasks)
        elif self.scenario_name == "Domain":
            self.scenario_tr = InstanceIncremental(dataset_train, nb_tasks=num_tasks)
            self.scenario_te = InstanceIncremental(dataset_test, nb_tasks=num_tasks)
        self.model = Model(num_classes=self.scenario_tr.nb_classes).cuda()

        self.num_tasks = num_tasks
        self.continuum = scenario

        self.eval_tr_loader = DataLoader(self.scenario_te[:], batch_size=264, shuffle=True, num_workers=6)

        self.eval_te_loader = DataLoader(self.scenario_te[:], batch_size=264, shuffle=True, num_workers=6)
        self.opt = optim.SGD(params=self.model.parameters(), lr=self.lr, momentum=0.9)

    def regularize_loss(self, model, loss):
        return loss

    def init_task(self, ind_task, task_set):
        data_loader_tr = DataLoader(task_set, batch_size=self.batch_size, shuffle=True, num_workers=6)
        if ind_task==0:
            # log before training
            self.init_log(ind_task_log=ind_task)
            self.test(ind_task_log=ind_task, train=False)
            self.test(ind_task_log=ind_task, data_loader=data_loader_tr, train=True)
            self.log_post_epoch_processing(0)
        return data_loader_tr

    def callback_task(self, ind_task, task_set):
        pass

    def test(self, ind_task_log, data_loader=None, train=False):

        if data_loader is None:
            data_loader = self.eval_te_loader
        for i_, (x_, y_, t_) in enumerate(data_loader):

            # data does not fit to the model if size<=1
            if x_.size(0) <= 1:
                continue

            y_ = y_.cuda()
            x_ = x_.cuda()

            self.model.eval()
            output = self.model(x_)
            loss = F.cross_entropy(output, y_)

            self.log_iter(ind_task_log, self.model, loss, output, y_, t_, train=train)

    def one_task_training(self, ind_task, data_loader):

        for epoch in range(self.nb_epochs):
            for i_, (x_, y_, t_) in enumerate(data_loader):

                # data does not fit to the model if size<=1
                if x_.size(0) <= 1:
                    continue

                y_ = y_.cuda()
                x_ = x_.cuda()

                self.model.train()
                self.opt.zero_grad()
                output = self.model(x_)

                assert output.shape[0] == y_.shape[0]
                loss = F.cross_entropy(output, y_)

                loss = self.regularize_loss(self.model, loss)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # clip gradient to avoid Nan
                self.opt.step()
                self.log_iter(ind_task+1, self.model, loss, output, y_, t_)

                if self.dev: break

            self.test(ind_task_log=ind_task+1)
            # we log and we print acc only for the last epoch
            self.log_post_epoch_processing(ind_task+1, print_acc=(epoch == self.nb_epochs - 1))
            if self.dev: break

        return

    def continual_training(self):

        for task_id, task_set in enumerate(self.scenario_tr):
            print("Task {}: Start".format(task_id))

            data_loader = self.init_task(task_id, task_set)
            self.init_log(task_id+1) # after init task!
            self.log_task(task_id, self.model)  # before training
            self.one_task_training(task_id, data_loader)
            self.callback_task(task_id, task_set)

        # last log (we log  at the beginning of each task except for the last one)

        self.log_task(self.num_tasks, self.model)
        self.post_training_log()
