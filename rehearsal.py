import torch
from torch import nn
from torch.nn import functional as F
from copy import deepcopy
from torch.utils import data
import numpy as np

import os
from copy import copy
from trainer import Trainer
from continuum.tasks.task_set import TaskSet
from memory import MemorySet


class Rehearsal(Trainer):
    def __init__(self, root_dir, dataset, scenario_name, model, num_tasks, dev):
        super().__init__(root_dir, dataset, scenario_name, model, num_tasks, dev)
        self.algo_name = "rehearsal"
        self.data_memory = None
        self.num_classes_per_task = 2
        self.nb_samples_rehearsal_per_class = 100
        self.samples_transfer = 5000
        self.sample_num = 100
        self.input_size = 1
        self.image_size = 28
        self.sample_dir = os.path.join(self.root_dir, "Samples")

    def init_task(self, ind_task: int, task_set: TaskSet):
        # ptit checkup
        if ind_task > 0:
            assert len(self.data_memory) == self.data_memory.nb_classes * self.nb_samples_rehearsal_per_class

    def callback_task(self, ind_task: int, task_set: TaskSet):
        if ind_task < self.num_tasks:
            self.manage_memory(ind_task, task_set)

        task_set.plot(self.sample_dir, f"training_{ind_task}.png",
                              nb_samples=100,
                              shape=[self.image_size, self.image_size, self.input_size])

    def manage_memory(self, ind_task: int, task_set: MemorySet):
        """
        Method to select samples for rehearsal
        :param ind_task: index of the task to select samples from
        :param nb_samples_rehearsal: number of samples saved per task
        :param samples_transfer: number of samples to incorporate in the new training set (samples_transfer > nb_samples_rehearsal)
        :return: updated train_set and test_set
        """
        nb_classes = task_set.nb_classes
        assert self.nb_samples_rehearsal_per_class * nb_classes < len(task_set._y)
        indexes = np.random.randint(0, len(task_set._y), self.nb_samples_rehearsal_per_class * nb_classes)
        samples, labels, task_ids = task_set.get_raw_samples(indexes)

        if self.data_memory is not None:
            self.data_memory.add_samples(samples, labels, task_ids)
        else:
            self.data_memory = MemorySet(samples, labels, task_ids, None)

        self.data_memory.plot(self.sample_dir, f"memory_{ind_task}.png",
                              nb_samples=100,
                              shape=[self.image_size, self.image_size, self.input_size])

    def one_task_training(self, ind_task: int, task_set: TaskSet):
        """
        In vanilla rehearsal we add memory data to training data and balance classes
        """

        # We convert task set to a memory set because we need to manipulate the list_ID to balance classes
        task_memory_set = None
        if ind_task > 0:
            task_memory_set = MemorySet(task_set._x,
                                        task_set._y,
                                        task_set._t,
                                        task_set.trsf,
                                        task_set.data_type)

            task_memory_set.concatenate(self.data_memory)
            task_memory_set.balance_classes()

            print("############  INFOS #############")
            print(task_memory_set.get_classes())
            print(len(task_memory_set))
        else:
            task_memory_set = task_set

        super().one_task_training(ind_task, task_memory_set)
