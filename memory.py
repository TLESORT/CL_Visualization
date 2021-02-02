from continuum.tasks.task_set import TaskSet
import numpy as np
from torchvision import transforms


class MemorySet(TaskSet):
    """
    A task set designed for Rehearsal strategies
    """

    def __init__(
            self,
            x: np.ndarray,
            y: np.ndarray,
            t: np.ndarray,
            trsf: transforms.Compose,
            data_type: str = "image_array"
    ):
        super().__init__(x=x, y=y, t=t, trsf=trsf, data_type=data_type)

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

    def get_random_raw_samples(self, nb_samples):
        nb_tot_samples = self._x.shape[0]
        indexes = np.random.randint(0, nb_tot_samples, nb_samples)
        return self.get_raw_samples(indexes)

    def get_nb_samples(self):
        """
        The nb of samples is the size of the labels vector
        """
        return len(self._y)

    def concatenate(self, other_memory_set):
        current_len = len(self)
        len_data = self.get_nb_samples()

        # we add the list of id from other_memory_set
        # we update the id value with the one in this memory
        dict_new_memory = {current_len + i: len_data + other_memory_set.list_IDs[i] for i in
                           range(len(other_memory_set))}

        self._x = np.concatenate((self._x, other_memory_set._x), axis=0)
        self._y = np.concatenate((self._y, other_memory_set._y), axis=0)
        self._t = np.concatenate((self._t, other_memory_set._t), axis=0)

        self.list_IDs.update(dict_new_memory)

    def increase_size(self, increase_factor):
        """
        artificially increase size of memory for balance purpose
        """
        assert increase_factor > 1.0
        current_len = len(self.list_IDs)
        new_len = int(current_len * increase_factor)

        # create dictionnary with new keys
        new_dic = {i: np.random.choice(len(self._y)) for i in range(current_len, new_len)}
        self.list_IDs.update(new_dic)

    def increase_size_class(self, increase_factor, class_label):
        """
        artificially increase size of memory for balance purpose
        """
        assert increase_factor > 1.0
        assert class_label in self.get_classes()

        nb_instance_class = self.get_nb_samples(class_label)
        nb_new_instance_needed = int(nb_instance_class * (increase_factor - 1))

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
        list_classes = np.unique(self._y)

        # first we get the number of samples for each class
        list_samples_per_classes = {}
        for _class in list_classes:
            nb_samples = self.get_nb_samples_class(_class)
            list_samples_per_classes[_class] = nb_samples

        max_samples = max(list_samples_per_classes, key=list_samples_per_classes.get)

        # we increase the nb of samples for classes under represented
        for _class in list_classes:
            nb_samples = list_samples_per_classes[_class]
            increase_factor = 1.0 * max_samples / nb_samples
            # we tolerate 5% error
            if increase_factor > 1.05:
                self.increase_size_class(increase_factor, _class)

    def balance_tasks(self):
        """
        modify list_ID so classes will be balanced while loaded with data loader
        """
        # TODO

    def get_nb_instances_class(self, class_label):
        """
        get the number of iteration of certain class
        """
        return sum(self._y[value] == class_label for value in self.list_IDs.values())

    def get_indexes_class(self, class_label):
        """
        get the number of iteration of certain class
        """
        return [value for value in self.list_IDs.values() if self._y[value] == class_label]

    def get_nb_samples_class(self, class_label):
        """
        get the number of samples of certain class.
         (it is different from get_ize_class because a single sample can be instanciated several times
         if its id is the ID_list several time)
        """
        return len(np.where(self._y == class_label))

    def get_nb_instances_task(self, task_id):
        """
        get the number of iteration of certain class
        """
        return sum(self._t[value] == task_id for value in self.list_IDs.values())

    def get_nb_samples_task(self, task_id):
        """
        get the number of samples of certain class.
         (it is different from get_ize_class because a single sample can be instanciated several times
         if its id is the ID_list several time)
        """
        return len(np.where(self._t == task_id))

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
