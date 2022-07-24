import torch
from torch import nn
from torch.nn import functional as F
from copy import deepcopy
from torch.utils.data import DataLoader
import numpy as np

import os
from copy import copy
from continuum.tasks.task_set import TaskSet
from continuum.tasks.base import TaskType
from memory import MemorySet

from Methods.trainer import Trainer

class Rehearsal(Trainer):
    def __init__(self, config):
        super().__init__(config)
        self.name_algo = "rehearsal"
        self.data_memory = None
        self.replay_balance = config.replay_balance
        self.nb_samples_rehearsal_per_class = config.nb_samples_rehearsal_per_class

    def sample_task(self, task_set):
        """
        Method to select samples for rehearsal
        :param task_set: current task set of data
        :return: updated train_set and test_set
        """
        nb_classes = task_set.nb_classes
        assert self.nb_samples_rehearsal_per_class * nb_classes < len(task_set._y), \
            f"{self.nb_samples_rehearsal_per_class} x {nb_classes} =" \
            f" {self.nb_samples_rehearsal_per_class * nb_classes} vs {len(task_set._y)} "
        indexes = np.random.randint(0, len(task_set._y), self.nb_samples_rehearsal_per_class * nb_classes)

        if task_set.data_type == TaskType.H5:
            unique_indexes, inverse_ids = np.unique(indexes, return_inverse=True)
            samples, labels, task_ids = task_set.get_raw_samples(unique_indexes)
            samples, labels = samples[inverse_ids], labels[inverse_ids]
            if task_ids is not None:
                task_ids = task_ids[inverse_ids]
        else:
            samples, labels, task_ids = task_set.get_raw_samples(indexes)

        assert task_ids is not None # if it is none it would be better to create a valid task id tensor

        return MemorySet(samples, labels, task_ids, None)

    def init_task(self, ind_task: int, task_set: TaskSet):


        if not self.data_encoded: # if data is encoded we can not plot it
            task_set.plot(self.sample_dir, f"training_{ind_task}.png",
                          nb_samples=100,
                          shape=[self.model.image_size, self.model.image_size, self.model.input_dim])

        samples_memory = self.sample_task(task_set)

        # add replay samples in taskset without the new samples
        task_memory_set = None
        if ind_task > 0:

            new_classes = task_set.get_classes()
            previous_classes = self.data_memory.get_classes()

            # ptit checkup
            if self.scenario_name == "Domain" or self.scenario_name == "SpuriousFeatures":
                assert len(self.data_memory) == self.data_memory.nb_classes *\
                       self.nb_samples_rehearsal_per_class *\
                       ind_task, \
                    f"{len(self.data_memory)} ==" \
                    f" {self.data_memory.nb_classes} *" \
                    f" {self.nb_samples_rehearsal_per_class} *" \
                    f"{ind_task}"
            else:
                assert len(self.data_memory) == self.data_memory.nb_classes * self.nb_samples_rehearsal_per_class, \
                    f"{len(self.data_memory)} == {self.data_memory.nb_classes} * {self.nb_samples_rehearsal_per_class}"
            # We convert task set to a memory set because we need to manipulate the list_ID to balance classes
            if task_set.data_type == TaskType.H5:
                samples, labels, task_ids = task_set.get_raw_samples(np.arange(len(task_set)))
                task_memory_set = MemorySet(samples,
                                            labels,
                                            task_ids,
                                            task_set.trsf)
            else:
                task_memory_set = MemorySet(task_set._x,
                                            task_set._y,
                                            task_set._t,
                                            task_set.trsf)

            task_memory_set.concatenate(self.data_memory)
            task_memory_set.balance_classes(ratio=self.replay_balance, new_classes=new_classes, previous_classes=previous_classes)
            task_memory_set.check_internal_state()
        else:
            task_memory_set = task_set

        if not self.data_encoded: # if data is encoded we can not plot it
            task_memory_set.plot(self.sample_dir, f"training_with_replay_{ind_task}.png",
                          nb_samples=100,
                          shape=[self.model.image_size, self.model.image_size, self.model.input_dim])

        # merge memory with new samples
        if self.data_memory is not None:
            self.data_memory.concatenate(samples_memory)
        else:
            self.data_memory = samples_memory

        if not self.data_encoded: # if data is encoded we can not plot it
            self.data_memory.plot(self.sample_dir, f"memory_{ind_task}.png",
                                  nb_samples=100,
                                  shape=[self.model.image_size, self.model.image_size, self.model.input_dim])

        if self.verbose and ind_task > 0:
            print("Composition of new dataset")
            print(len(task_memory_set))
            print(len(task_memory_set._y))
            for class_ in task_memory_set.get_classes():
                print(f"{class_} : {len(np.where(task_memory_set._y[list(task_memory_set.list_IDs.values())] == class_)[0])}")

        return super().init_task(ind_task, task_memory_set)
