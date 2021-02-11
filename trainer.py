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
from continuum import ClassIncremental
from continuum import Rotations

from eval import Continual_Evaluation


class Trainer(Continual_Evaluation):
    def __init__(self, args, root_dir, scenario_name, num_tasks, verbose, dev):

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
        self.dev = dev
        self.nb_epochs = args.nb_epochs

        dataset_train = get_dataset(self.dir_data, args.dataset, train=True)
        dataset_test = get_dataset(self.dir_data, args.dataset, train=False)

        scenario = None
        if scenario_name == "Rotations":
            self.scenario_tr = Rotations(dataset_train, nb_tasks=num_tasks)
        elif scenario_name == "Disjoint":
            self.scenario_tr = ClassIncremental(dataset_train, nb_tasks=num_tasks)

        self.model = Model(num_classes=self.scenario_tr.nb_classes).cuda()

        self.num_tasks = num_tasks
        self.continuum = scenario

        fisher_set = None
        if scenario_name == "Rotations":
            fisher_set = Rotations(dataset_train, nb_tasks=1)
        elif scenario_name == "Disjoint":
            fisher_set = ClassIncremental(dataset_train, nb_tasks=1)  # .sub_sample(1000)
            self.test_set = ClassIncremental(dataset_test, nb_tasks=1)

        #self.fisher_eval_loader = DataLoader(fisher_set[:], batch_size=264, shuffle=True, num_workers=6)
        self.eval_tr_loader = DataLoader(self.test_set[:], batch_size=264, shuffle=True, num_workers=6)

        self.eval_te_loader = DataLoader(self.test_set[:], batch_size=264, shuffle=True, num_workers=6)
        self.opt = optim.SGD(params=self.model.parameters(), lr=self.lr, momentum=0.9)

    def regularize_loss(self, model, loss):
        return loss

    def init_task(self, ind_task, task_set):
        return task_set

    def callback_task(self, ind_task, task_set):
        pass

    def test(self):
        correct = 0.0
        classe_prediction = np.zeros(self.scenario_tr.nb_classes)
        classe_total = np.zeros(self.scenario_tr.nb_classes)
        classe_wrong = np.zeros(self.scenario_tr.nb_classes)  # Images wrongly attributed to a particular class
        for i_, (x_, y_, _) in enumerate(self.eval_te_loader):

            # data does not fit to the model if size<=1
            if x_.size(0) <= 1:
                continue

            y_ = y_.cuda()
            x_ = x_.cuda()

            self.model.eval()
            output = self.model(x_)

            correct += (output.max(dim=1)[1] == y_).data.sum()
            pred = output.data.max(1, keepdim=True)[1]
            for i in range(y_.shape[0]):
                if pred[i].detach().cpu()[0] == y_[i].detach().cpu():
                    classe_prediction[pred[i].detach().cpu()[0]] += 1
                else:
                    classe_wrong[pred[i].detach().cpu()[0]] += 1

                classe_total[y_[i]] += 1

        print("Test Accuracy: {} %".format(100.0 * correct / len(self.eval_te_loader.dataset)))

        if self.verbose:
            for i in range(self.scenario_tr.nb_classes):
                print("Task " + str(i) + "- Prediction :" + str(
                    classe_prediction[i] / classe_total[i]) + "% - Total :" + str(
                    classe_total[i]) + "- Wrong :" + str(classe_wrong[i]))

    def one_task_training(self, ind_task, task_set):
        correct = 0

        accuracy_per_epoch = []

        data_loader = DataLoader(task_set, batch_size=self.batch_size, shuffle=True, num_workers=6)
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
                loss = F.cross_entropy(output, y_)

                loss = self.regularize_loss(self.model, loss)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0) # clip gradient to avoid Nan
                self.opt.step()
                self.log_iter(ind_task, self.model, loss)

                assert output.shape[0] == y_.shape[0]
                correct += (output.max(dim=1)[1] == y_).data.sum()
                if self.dev: break
            if self.dev: break

            accuracy_per_epoch.append(1.0 * correct / len(data_loader))

        return

    def continual_training(self):

        for task_id, task_set in enumerate(self.scenario_tr):
            print("Task {}: Start".format(task_id))
            # log is disabled for first debugging steps
            self.init_log(task_id)

            task_set = self.init_task(task_id, task_set)
            # log is disabled for first debugging steps
            self.log_task(task_id, self.model)  # before training
            self.one_task_training(task_id, task_set)
            self.callback_task(task_id, task_set)

            self.test()

        # last log (we log  at the beginning of each task except for the last one)

        self.init_log(self.num_tasks)
        self.log_task(self.num_tasks, self.model)
        self.post_training_log()
