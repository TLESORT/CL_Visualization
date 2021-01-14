import torch
from torch import nn
from torch.nn import functional as F
from copy import deepcopy
from torch.utils import data

import os
from copy import copy
from trainer import Trainer
#from Continual_Learning_Data_Former.continuum.continuum_loader import ContinuumSetLoader


class Rehearsal(Trainer):
    def __init__(self, scenario, continuum, model):
        super().__init__(scenario, continuum, model)
        self.model = model
        self.continuum = continuum
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

        path = os.path.join(self.sample_dir, 'training_' + str(ind_task) + '.png')
        self.continuum.visualize_sample(path, self.sample_num, [self.image_size, self.image_size, self.input_size])

    def callback_task(self, ind_task):
        pass

    def manage_memory(self, ind_task, nb_samples_rehearsal, samples_transfer):
        """
        Method to select samples for rehearsal
        :param ind_task: index of the task to select samples fro;
        :param nb_samples_rehearsal: number of samples saved per task
        :param samples_transfer: number of samples to incorporate in the new training set (samples_transfer > nb_samples_rehearsal)
        :return: updated train_set and test_set
        """
        # save sample before modification of training set
        self.continuum.set_task(ind_task)
        x_tr, y_tr = self.continuum.get_sample(nb_samples_rehearsal, shape=[self.input_size, self.image_size, self.image_size])
        self.task_samples = copy(x_tr).reshape(-1, self.input_size* self.image_size* self.image_size)
        self.task_labels = copy(y_tr)

        # create data loader with memory from previous task
        if ind_task > 0:
            # balanced the number of sample and incorporate it in the memory
            self.continuum.set_task(ind_task)
            # put the memory inside the training dataset
            self.continuum.concatenate(self.data_memory)
            self.continuum.set_task(ind_task)
            self.continuum.shuffle_task()

        # Add data to memory at the end
        c1 = 0
        c2 = 1
        tasks_tr = []  # reset the list

        # save samples from the actual task in the memory

        tasks_tr.append([(c1, c2), self.task_samples, self.task_labels])
        increase_factor = int(samples_transfer / self.nb_samples_rehearsal) * self.num_classes_per_task

        assert increase_factor > 0

        if ind_task == 0:
            self.data_memory = ContinuumSetLoader(tasks_tr, transform=None, load_images=False)
            self.data_memory.increase_size(increase_factor)
        else:
            new_data = ContinuumSetLoader(tasks_tr, transform=None, load_images=False)
            new_data.increase_size(increase_factor)
            self.data_memory.concatenate(new_data)

        path = os.path.join(self.sample_dir, 'memory_' + str(ind_task) + '.png')
        self.data_memory.visualize_sample(path, self.sample_num, [self.image_size, self.image_size, self.input_size])


