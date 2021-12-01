from typing import Tuple, Union

import numpy as np
from torchvision import transforms

from continuum.tasks.task_set import ArrayTaskSet, TaskType


class MemorySet(ArrayTaskSet):
    """
    A task set designed for Rehearsal strategies
    """

    def __init__(
            self,
            x: np.ndarray,
            y: np.ndarray,
            t: np.ndarray,
            trsf: transforms.Compose
    ):
        super().__init__(x=x, y=y, t=t, trsf=trsf, target_trsf=None)
        self.data_type = TaskType.IMAGE_ARRAY

        list_labels_id = range(len(self._y))

        # dictionnary used by pytorch loader
        self.list_IDs = {i: list_labels_id[i] for i in range(0, len(list_labels_id))}

    def reset_list_IDs(self):
        list_labels_id = range(len(self._y))
        self.list_IDs = {i: list_labels_id[i] for i in range(0, len(list_labels_id))}

    def __len__(self):
        """
        The size of a memory depend of the list
        """
        return len(self.list_IDs)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int, int]:
        """Method used by PyTorch's DataLoaders to query a sample and its target."""
        # convert instance id into data id
        data_id = self.list_IDs[index]
        assert data_id <= len(self._y), f"data id {data_id} vs taille taskset {len(self._y)}"
        return super().__getitem__(data_id)

    def get_random_raw_samples(self, nb_samples):
        nb_tot_samples = self._x.shape[0]
        indexes = np.random.randint(0, nb_tot_samples, nb_samples)
        return self.get_raw_samples(indexes)

    def get_nb_samples(self):
        """
        The nb of samples is the size of the labels vector
        """
        return len(self._y)

    def add_samples(self, x: np.ndarray, y: np.ndarray, t: Union[None, np.ndarray] = None, list_IDs=None):
        """Add memory for rehearsal.

        :param x: Sampled data chosen for rehearsal.
        :param y: The associated targets of `x_memory`.
        :param t: The associated task ids. If not provided, they will be
                         defaulted to -1.
        :param dict_new_memory: dict with indexes of samples
        """

        current_len = len(self)
        len_new_data = len(y)
        len_data = len(self._y)

        dict_new_memory = None
        if list_IDs is None:
            dict_new_memory = {current_len + i: len_data + i for i in range(len_new_data)}
        else:
            dict_new_memory = {current_len + i: len_data + list_IDs[i] for i in
                               range(len(list_IDs))}

        super().add_samples(x, y, t)

        self.list_IDs.update(dict_new_memory)

    def concatenate(self, other_memory_set):
        """
        This function is done to concatenate two memory sets
        """
        # Sanity Check
        self.check_internal_state()
        other_memory_set.check_internal_state()

        # start merge
        self.add_samples(other_memory_set._x,
                         other_memory_set._y,
                         other_memory_set._t,
                         other_memory_set.list_IDs)

    def increase_size(self, increase_factor):
        """
        artificially increase size of memory for balance purpose
        """
        assert increase_factor > 1.0
        current_len = len(self.list_IDs)
        new_len = int(round(current_len * increase_factor))

        # create dictionnary with new keys
        new_dic = {i: np.random.choice(len(self._y)) for i in range(current_len, new_len)}
        self.list_IDs.update(new_dic)

    def increase_size_class(self, increase_factor, class_label):
        """
        artificially increase size of memory for balance purpose
        """
        assert increase_factor > 1.0
        assert class_label in self.get_classes()

        nb_instance_class = self.get_nb_instances_class(class_label)
        nb_new_instance_needed = int(round(nb_instance_class * increase_factor)) - nb_instance_class

        len_list = len(self)

        class_indexes = self.get_indexes_class(class_label)

        # create dictionnary with new keys
        new_dic = {i: np.random.choice(class_indexes) for i in range(len_list, len_list + nb_new_instance_needed)}
        self.list_IDs.update(new_dic)

    def balance_classes(self):
        """
        modify list_ID so classes will be balanced while loaded with data loader
        """
        self.reset_list_IDs()
        list_classes = self.get_classes()

        # first we get the number of samples for each class
        list_samples_per_classes = {}
        for _class in list_classes:
            nb_samples = self.get_nb_samples_class(_class)
            list_samples_per_classes[_class] = nb_samples

        ind_max_samples = max(list_samples_per_classes, key=list_samples_per_classes.get)
        max_samples = list_samples_per_classes[ind_max_samples]

        # we increase the nb of samples for classes under represented
        for _class in list_classes:
            nb_samples = list_samples_per_classes[_class]
            assert nb_samples <= max_samples
            increase_factor = 1.0 * max_samples / nb_samples
            # we tolerate 5% error
            if increase_factor > 1.05:
                self.increase_size_class(increase_factor, _class)

        # Check if everything looks good
        self.check_internal_state()

    def balance_tasks(self):
        """
        modify list_ID so classes will be balanced while loaded with data loader
        """
        # TODO

    def get_nb_instances_class(self, class_label):
        """
        get the number of iteration of certain class
        """
        return sum(self._y[self.list_IDs[value]] == class_label for value in self.list_IDs.values())

    def get_indexes_class(self, class_label):
        """
        get the number of iteration of certain class
        """
        return [value for value in self.list_IDs.values() if self._y[value] == class_label]


    def get_indexes_task(self, task_label):
        """
        get the number of iteration of certain task
        """
        return [value for value in self.list_IDs.values() if self._t[value] == task_label]

    def get_nb_samples_class(self, class_label):
        """
        get the number of samples of certain class.
         (it is different from get_ize_class because a single sample can be instanciated several times
         if its id is the ID_list several time)
        """
        return len(np.where(self._y == class_label)[0])

    def get_nb_instances_task(self, task_id):
        """
        get the number of iteration of certain class
        """
        return sum(self._t[self.list_IDs[value]] == task_id for value in self.list_IDs.values())

    def get_nb_samples_task(self, task_id):
        """
        get the number of samples of certain class.
         (it is different from get_ize_class because a single sample can be instanciated several times
         if its id is the ID_list several time)
        """
        return len(np.where(self._t == task_id)[0])

    def reduce_size(self):
        """
        reduce the number of the memory size.
         If samples id are redundant in the ID_list it will remove redundancy then,
         it will delete some samples if the reduction is to big.
        """
        # TODO
        pass

    def reduce_size_class(self, reduction_factor):
        """
        reduce the number of a certain class.
         If samples id are redundant in the ID_list it will remove redundancy then,
         it will delete some samples if the reduction is to big.
        """
        # TODO
        pass

    def reduce_size_task(self, reduction_factor):
        """
        reduce the number of a certain task.
         If samples id are redundant in the ID_list it will remove redundancy then,
         it will delete some samples if the reduction is to big.
        """
        # TODO
        pass

    def delete_tasks_samples(self, task_id):
        """
        remove all examples of a specified task.
        """
        # TODO
        pass

    def delete_class_samples(self, class_id):
        """
        remove all examples of a specified class.
        """
        # TODO
        pass

    def check_internal_state(self):

        assert len(self._x) == len(self._y)
        assert len(self._y) == len(self._t)
        nb_samples = len(self._y)
        nb_instances = len(self.list_IDs)

        assert nb_instances >= nb_samples

        for key, id_value in self.list_IDs.items():
            assert id_value < nb_samples
            assert key < nb_instances

    def get_random_samples(self, nb_samples):
        nb_tot_instances = len(self)
        indexes = np.random.randint(0, nb_tot_instances, nb_samples)
        return self.get_samples(indexes)
