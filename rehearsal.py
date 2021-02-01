import torch
from torch import nn
from torch.nn import functional as F
from copy import deepcopy
from torch.utils import data

import os
from copy import copy
from trainer import Trainer
from continuum.tasks.task_set import TaskSet
from memory import MemorySet


class Rehearsal(Trainer):
    def __init__(self, root_dir, dataset, scenario_name, model, num_tasks):
        super().__init__(root_dir, dataset, scenario_name, model, num_tasks)
        self.algo_name = "rehearsal"
        self.data_memory = None
        self.num_classes_per_task = 2
        self.nb_samples_rehearsal = 100
        self.samples_transfer = 5000
        self.sample_num = 100
        self.input_size = 1
        self.image_size = 28
        self.sample_dir = os.path.join(self.root_dir , "Samples")


    def init_task(self, ind_task):
        if ind_task < self.num_tasks:
            self.manage_memory(ind_task, self.nb_samples_rehearsal, self.samples_transfer)

        self.scenario_tr[ind_task].plot(self.sample_dir, f"training_{ind_task}.png",
                                      nb_samples= 100,
                                      shape = [self.image_size, self.image_size, self.input_size])

    def callback_task(self, ind_task):
        pass

    def manage_memory(self, ind_task, nb_samples_rehearsal, samples_transfer):
        """
        Method to select samples for rehearsal
        :param ind_task: index of the task to select samples from
        :param nb_samples_rehearsal: number of samples saved per task
        :param samples_transfer: number of samples to incorporate in the new training set (samples_transfer > nb_samples_rehearsal)
        :return: updated train_set and test_set
        """
        # save sample before modification of training set
        self.scenario_tr.set_task(ind_task)
        x_tr, y_tr = self.scenario_tr.get_sample(nb_samples_rehearsal, shape=[self.input_size, self.image_size, self.image_size])
        self.task_samples = copy(x_tr).reshape(-1, self.input_size* self.image_size* self.image_size)
        self.task_labels = copy(y_tr)

        # create data loader with memory from previous task
        if ind_task > 0:
            # put the memory inside the training dataset
            self.scenario_tr.add_samples(self.data_memory)

        increase_factor = int(samples_transfer / self.nb_samples_rehearsal) * self.num_classes_per_task
        assert increase_factor > 0

        if ind_task == 0:
            self.data_memory = MemorySet(self.task_samples, self.task_labels)
            #self.data_memory.increase_size(increase_factor)
        else:
            #new_data = TaskSet(self.task_samples, self.task_label)
            #new_data.increase_size(increase_factor)
            self.data_memory.add_samples(self.task_samples, self.task_label)

        self.data_memory.plot(self.sample_dir, f"memory_{ind_task}.png",
                                      nb_samples= 100,
                                      shape = [self.image_size, self.image_size, self.input_size])



