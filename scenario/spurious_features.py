import warnings
from typing import Callable, List, Union

import numpy as np
import torch
from torchvision import transforms

from continuum.datasets import _ContinuumDataset
from continuum.tasks import TaskSet

from continuum.scenarios import InstanceIncremental
from copy import copy


class SpuriousFeatures(InstanceIncremental):
    """Continual Loader, generating datasets for the consecutive tasks.

    Scenario: SpuriousFeatures scenarios, use same data for all task but with spurious correlation into data different at each task.
    Each task get a specific permutation, such as all tasks are different.

    :param cl_dataset: A continual dataset.
    :param nb_tasks: The scenario's number of tasks.
    :param base_transformations: List of transformations to apply to all tasks.
    :param seed: initialization seed for the random number generator.
    :param correlation: correlation between label and spurious feature, 1.0 full correlation, 0.0 no correlation.
     To lower correlation we just do not add spurious feature to the data point.
    """

    def __init__(
            self,
            cl_dataset: _ContinuumDataset,
            nb_tasks: int = None,
            base_transformations: List[Callable] = None,
            seed: int = 0,
            correlation: float = 1.0,
            train=True
    ):
        assert nb_tasks is not None

        self.training = train
        self.correlation = correlation
        super().__init__(
            cl_dataset=cl_dataset,
            nb_tasks=nb_tasks,
            transformations=base_transformations,
            random_seed = seed
        )
        self.image_size = 32

    @property
    def nb_tasks(self) -> int:
        """Number of tasks in the whole continual setting."""

        return self._nb_tasks

    def _generate_two_colors(self, ind_task):
        rng = np.random.RandomState(seed=ind_task)
        colors = rng.choice(range(16), size=(2, 3)) * 16  # 16 * 16 = 256, we don't want to have too close colors

        return colors

    def class_remapping(self, y):
        return y // 5

    def _data_transformation(self, x, labels, ind_task):
        """"Transform data for scenario purposes
         it can not be done with classical transform since the transformation depends on the label
         We convert [1,28,28] images into [3,28,28]
        """
        nb_samples = len(labels)

        # We convert [1,28,28] images into [3,28,28]
        # data_task = np.zeros((nb_samples, self.image_size, self.image_size, 3))
        # data_task[:, :, :, 0] = x
        # data_task[:, :, :, 1] = x
        # data_task[:, :, :, 2] = x

        data_task = copy(x)

        if (self.training or (ind_task < self.nb_tasks - 1)):

            centers = torch.randint(2, self.image_size - 2, (nb_samples, 2))

            colors = self._generate_two_colors(ind_task)
            for i in range(nb_samples):
                # uniform sampling between 0 and 1 to create correlation
                if np.random.uniform() < self.correlation:
                    square_size = 2  # random.choice([1, 2, 3])
                    center = centers[i, :]  # [5, 5]
                    label = labels[i]
                    color = colors[label]
                    x_min = max(0, center[0] - square_size)
                    x_max = min(self.image_size - 1, center[0] + square_size)
                    y_min = max(0, center[1] - square_size)
                    y_max = min(self.image_size - 1, center[1] + square_size)

                    data_task[i, x_min:x_max, y_min:y_max, 0] = color[0]
                    data_task[i, x_min:x_max, y_min:y_max, 1] = color[1]
                    data_task[i, x_min:x_max, y_min:y_max, 2] = color[2]

        return data_task

    # nothing to do in the setup function
    def _setup(self, nb_tasks: int) -> int:
        x, y, t = self.cl_dataset.get_data()
        y = self.class_remapping(y)
        self.dataset = [x, y, t]

        if not self.training:
            # for test we add one more task with data without modif
            nb_tasks += 1

        return nb_tasks

    def _select_data_by_task(
            self,
            task_index: Union[int, slice, np.ndarray]
    ) -> Union[np.ndarray, np.ndarray, np.ndarray, Union[int, List[int]]]:
        """Selects a subset of the whole data for a given task.

        This class returns the "task_index" in addition of the x, y, t data.
        This task index is either an integer or a list of integer when the user
        used a slice. We need this variable when in segmentation to disangle
        samples with multiple task ids.

        :param task_index: The unique index of a task. As for List, you can use
                           indexing between [0, len], negative indexing, or
                           even slices.
        :return: A tuple of numpy array being resp. (1) the data, (2) the targets,
                 (3) task ids, and (4) the actual task required by the user.
        """
        assert task_index < self.nb_tasks

        x, y, t = self.dataset

        x = self._data_transformation(x, y, task_index)
        t = torch.ones(y.shape[0]) * task_index
        return x, y, t, task_index, np.arange(y.shape[0])


""""
def generate_SQOLOR(path_dir, dataset, name_scenario, train=True):
    NB_TASKS = 2

    if dataset == "SQOLOR":
        if train:
            nb_samples = 10000
        else:
            nb_samples = 1000
        data_vector = np.zeros((nb_samples, 3, 28, 28))
        labels = np.random.randint(0, 2, nb_samples)
    else:
        dataset_name = dataset.replace("SQOLOR_", "")
        dataset = get_dataset(path_dir, dataset_name, name_scenario, train=train)
        x, y, t = dataset.get_data()
        nb_samples = y.shape[0]
        labels = y // 5
        data_vector = np.zeros((nb_samples, 3, 28, 28))
        data_vector[:, 0, :, :] = x
        data_vector[:, 1, :, :] = x
        data_vector[:, 2, :, :] = x

    if train:
        full_data = np.zeros((0, 3, 28, 28))
        full_label = np.zeros(0)
        full_task_id = np.zeros(0)
        for j in range(NB_TASKS):
            if j == 0:
                rng = np.random.RandomState(seed=1)
                # for first tasks we only learn on spurious features to force model to be wrong after
                data_task = np.zeros((nb_samples, 3, 28, 28))
            else:
                rng = np.random.RandomState(seed=j)
                data_task = copy(data_vector)
            colors = rng.choice(range(16), size=(2, 3)) * 16
            centers = torch.randint(0, 28, (nb_samples, 2))

            for i in range(nb_samples):
                square_size = 2 #random.choice([1, 2, 3])
                center = centers[i, :] # [5, 5]
                label = labels[i]
                color = colors[label]
                x_min = max(0, center[0] - square_size)
                x_max = min(27, center[0] + square_size)
                y_min = max(0, center[1] - square_size)
                y_max = min(27, center[1] + square_size)

                data_task[i, 0, x_min:x_max, y_min:y_max] = color[0]
                data_task[i, 1, x_min:x_max, y_min:y_max] = color[1]
                data_task[i, 2, x_min:x_max, y_min:y_max] = color[2]

            task_id = np.ones(len(labels)) * j

            full_data = np.concatenate((full_data, data_task), axis=0)
            full_label = np.concatenate((full_label, labels), axis=0)
            full_task_id = np.concatenate((full_task_id, task_id), axis=0)

    else:
        full_data = data_vector
        full_label = labels
        full_task_id = np.zeros(len(labels))

    cl_dataset = InMemoryDataset(torch.FloatTensor(full_data), torch.LongTensor(full_label), full_task_id,
                                 data_type="tensor")
    return cl_dataset

"""
