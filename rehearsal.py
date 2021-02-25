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
    def __init__(self, args, root_dir, scenario_name, num_tasks, verbose, dev):
        super().__init__(args, root_dir, scenario_name, num_tasks, verbose, dev)
        self.algo_name = "rehearsal"
        self.data_memory = None
        self.num_classes_per_task = 2
        self.nb_samples_rehearsal_per_class = 100
        self.samples_transfer = 5000
        self.sample_num = 100
        self.input_size = 1
        self.image_size = 28

    def callback_task(self, ind_task: int, task_set: TaskSet):
        pass

    def sample_task(self, task_set):
        """
                Method to select samples for rehearsal
                :param ind_task: index of the task to select samples from
                :param nb_samples_rehearsal: number of samples saved per task
                :param samples_transfer: number of samples to incorporate in the new training set (samples_transfer > nb_samples_rehearsal)
                :return: updated train_set and test_set
                """
        nb_classes = task_set.nb_classes
        assert self.nb_samples_rehearsal_per_class * nb_classes < len(task_set._y), \
            f"{self.nb_samples_rehearsal_per_class} x {nb_classes} =" \
            f" {self.nb_samples_rehearsal_per_class * nb_classes} vs {len(task_set._y)} "
        indexes = np.random.randint(0, len(task_set._y), self.nb_samples_rehearsal_per_class * nb_classes)
        samples, labels, task_ids = task_set.get_raw_samples(indexes)

        return MemorySet(samples, labels, task_ids, None)

    def init_task(self, ind_task: int, task_set: TaskSet):

        task_set.plot(self.sample_dir, f"training_{ind_task}.png",
                      nb_samples=100,
                      shape=[self.image_size, self.image_size, self.input_size])

        samples_memory = self.sample_task(task_set)

        # add replay samples in taskset without the new samples
        task_memory_set = None
        if ind_task > 0:
            # ptit checkup
            assert len(self.data_memory) == self.data_memory.nb_classes * self.nb_samples_rehearsal_per_class, \
                f"{len(self.data_memory)} == {self.data_memory.nb_classes} * {self.nb_samples_rehearsal_per_class}"
            # We convert task set to a memory set because we need to manipulate the list_ID to balance classes
            task_memory_set = MemorySet(task_set._x,
                                        task_set._y,
                                        task_set._t,
                                        task_set.trsf,
                                        task_set.data_type)

            task_memory_set.concatenate(self.data_memory)
            task_memory_set.balance_classes()
            task_memory_set.check_internal_state()
        else:
            task_memory_set = task_set

        task_memory_set.plot(self.sample_dir, f"training_with_replay_{ind_task}.png",
                      nb_samples=100,
                      shape=[self.image_size, self.image_size, self.input_size])

        # merge memory with new samples
        if self.data_memory is not None:
            self.data_memory.concatenate(samples_memory)
        else:
            self.data_memory = samples_memory

        self.data_memory.plot(self.sample_dir, f"memory_{ind_task}.png",
                              nb_samples=100,
                              shape=[self.image_size, self.image_size, self.input_size])

        return task_memory_set
