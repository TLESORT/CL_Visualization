import torch
from torch.utils.data import DataLoader
from nngeometry.metrics import FIM
import torch.optim as optim
import torch.nn.functional as F
import pickle
import os
import numpy as np
from copy import deepcopy

from continuum import ClassIncremental
from continuum import Rotations
from continuum.datasets import MNIST


from eval import Continual_Evaluation


class Trainer(Continual_Evaluation):
    def __init__(self, scenario, continuum, model):

        self.root_dir = "./Archives"
        self.dir_data = os.path.join(self.root_dir, "Data")
        self.log_dir = os.path.join(self.root_dir , "Logs", scenario)
        self.num_tasks = continuum.nb_tasks
        self.model = model
        self.algo_name = "baseline"
        self.continuum = continuum

        dataset_train = MNIST("../Datasets/", download=True, train=True)
        dataset_test = MNIST("../Datasets/", download=True, train=False)

        fisher_set = None
        if scenario == "Rotations":
            fisher_set = Rotations(dataset_train, nb_tasks=1)
        elif scenario == "Disjoint":
            fisher_set = ClassIncremental(dataset_train, nb_tasks=1)#.sub_sample(1000)
            self.test_set = ClassIncremental(dataset_test, nb_tasks=1)

        self.eval_tr_loader = DataLoader(fisher_set, batch_size=25, shuffle=False, num_workers=6)
        self.eval_te_loader = DataLoader(self.test_set, batch_size=25, shuffle=False, num_workers=6)
        self.opt = optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9)



    def regularize_loss(self, model, loss):
        return loss

    def init_task(self, ind_task):
        pass

    def callback_task(self, ind_task):
        pass

    def test(self):
        correct = 0.0
        classe_prediction = np.zeros(10)
        classe_total = np.zeros(10)
        classe_wrong = np.zeros(10)  # Images wrongly attributed to a particular class
        for i_, (x_, t_) in enumerate(self.eval_te_loader):

            # data does not fit to the model if size<=1
            if x_.size(0) <= 1:
                continue

            t_ = t_.cuda()
            x_ = x_.cuda()

            self.model.eval()
            output = self.model(x_)
            #loss = F.cross_entropy(output, t_)

            #loss = self.regularize_loss(self.model, loss)
            correct += (output.max(dim=1)[1] == t_).data.sum()
            pred = output.data.max(1, keepdim=True)[1]
            for i in range(t_.shape[0]):
                if pred[i].cpu()[0] == t_[i].cpu():
                    classe_prediction[pred[i].cpu()[0]] += 1
                else:
                    classe_wrong[pred[i].cpu()[0]] += 1

                classe_total[t_[i]] += 1

        print(len(self.test_set))
        print("Test Accuracy: {} %".format(100*(1.0*correct)/len(self.test_set)))

        # for i in range(10):
        #     print("Task " + str(i) + "- Prediction :" + str(
        #         classe_prediction[i] / classe_total[i]) + "% - Total :" + str(
        #         classe_total[i]) + "- Wrong :" + str(classe_wrong[i]))


    def one_task_training(self, train_loader, ind_task):
        correct = 0
        for i_, (x_, y_, t_) in enumerate(train_loader):

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
            self.opt.step()
            self.log_iter(ind_task, self.model, loss)

            assert output.shape[0] == y_.shape[0]
            correct += (output.max(dim=1)[1] == y_).data.sum()

    def continual_training(self, scenario_tr):

        for task_id, dataset_tr in enumerate(scenario_tr):
            print("Task {}: Start".format(task_id))
            # log is disabled for first debugging steps
            #self.init_log(task_id)

            self.init_task(task_id)
            # log is disabled for first debugging steps
            #self.log_task(task_id, self.model) # before training
            self.one_task_training(dataset_tr, task_id)
            self.callback_task(task_id)

            self.test()


        # last log (we log  at the beginning of each task exept for the last one)

        # log is disabled for first debugging steps
        #self.init_log(self.num_tasks)
        #self.log_task(self.num_tasks, self.model)
        #self.post_training_log()



