import numpy as np
from memory import MemorySet
from torch.utils import data
import pytest

SIZE_MEMORY = 20


def gen_data():
    x_1 = np.random.randint(0, 255, size=(SIZE_MEMORY, 32, 32, 3))
    # label unbalanced of size $SIZE_MEMORY
    y_1 = [0, 0, 0, 0, 0,
           0, 0, 1, 1, 1,
           2, 2, 2, 2, 2,
           3, 4, 4, 5, 5]
    # for i in range(int(SIZE_MEMORY / 2)):
    #     y_1.append(np.ones(2) * i)
    y_1 = np.array(y_1)
    t_1 = np.ones(len(y_1))

    x_2 = np.random.randint(0, 255, size=(SIZE_MEMORY, 32, 32, 3))
    y_2 = np.copy(y_1) + 10
    t_2 = np.ones(len(y_2)) * 2

    assert x_1.shape[0] == len(y_1) == len(t_1)
    assert x_2.shape[0] == len(y_2) == len(t_2)

    return (x_1, y_1, t_1), (x_2, y_2, t_2)


def test_taskset():
    (x_1, y_1, t_1), _ = gen_data()

    memory_set = MemorySet(x_1, y_1, t_1, None)

    assert len(memory_set) == SIZE_MEMORY
    assert memory_set.get_nb_samples() == SIZE_MEMORY


@pytest.mark.parametrize("increase_factor", [2, 2.5, 5.07])
def test_increase_memory(increase_factor):
    (x_1, y_1, t_1), _ = gen_data()

    memory_set = MemorySet(x_1, y_1, t_1, None)

    before_len = len(memory_set)
    before_nb_samples = memory_set.get_nb_samples()

    memory_set.increase_size(increase_factor=increase_factor)

    assert len(memory_set) == int(round(before_len * increase_factor))
    # the number of samples stay the same only the id list grows
    assert memory_set.get_nb_samples() == before_nb_samples


def test_simple_concatenate():
    (x_1, y_1, t_1), (x_2, y_2, t_2) = gen_data()

    memory_set_1 = MemorySet(x_1, y_1, t_1, None)
    memory_set_2 = MemorySet(x_2, y_2, t_2, None)

    classes_1 = memory_set_1.get_classes()
    classes_2 = memory_set_2.get_classes()

    len_1 = len(memory_set_1)
    len_2 = len(memory_set_2)

    memory_set_1.concatenate(other_memory_set=memory_set_2)

    assert len(memory_set_1) == len_1 + len_2

    cat_classes = memory_set_1.get_classes()
    for _class in (list(classes_1) + list(classes_2)):
        assert _class in cat_classes, f"{cat_classes} vs {classes_1} + {classes_2}"


@pytest.mark.parametrize("increase_factors", [[1.6, 1], [1.5, 2.5], [1, 3], [1.75, 2.25]])
def test_concatenate_with_size_increase(increase_factors):
    (x_1, y_1, t_1), (x_2, y_2, t_2) = gen_data()
    fact1, fact2 = increase_factors

    memory_set_1 = MemorySet(x_1, y_1, t_1, None)
    memory_set_2 = MemorySet(x_2, y_2, t_2, None)

    if fact1 > 1.0: memory_set_1.increase_size(increase_factor=fact1)
    if fact2 > 1.0: memory_set_2.increase_size(increase_factor=fact2)

    classes_1 = memory_set_1.get_classes()
    classes_2 = memory_set_2.get_classes()

    instances_1 = memory_set_1.get_nb_samples()
    instances_2 = memory_set_2.get_nb_samples()

    len_1 = len(memory_set_1)
    len_2 = len(memory_set_2)

    memory_set_1.concatenate(other_memory_set=memory_set_2)

    assert len(memory_set_1) == len_1 + len_2
    assert memory_set_1.get_nb_samples() == instances_1 + instances_2

    cat_classes = memory_set_1.get_classes()
    for _class in (list(classes_1) + list(classes_2)):
        assert _class in cat_classes, f"{cat_classes} vs {classes_1} + {classes_2}"


def test_loader():
    (x_1, y_1, t_1), _ = gen_data()
    memory_set = MemorySet(x_1, y_1, t_1, None)
    train_loader = data.DataLoader(memory_set, batch_size=64, shuffle=True, num_workers=6)

    assert len(train_loader.dataset) == SIZE_MEMORY


@pytest.mark.parametrize("increase_factor", [3, 3.5, 1.75])
def test_loader_and_increase_factor(increase_factor):
    (x_1, y_1, t_1), _ = gen_data()

    memory_set = MemorySet(x_1, y_1, t_1, None)

    memory_set.increase_size(increase_factor=increase_factor)

    train_loader = data.DataLoader(memory_set, batch_size=64, shuffle=True, num_workers=6)

    assert len(train_loader.dataset) == SIZE_MEMORY * increase_factor


@pytest.mark.parametrize("increase_factor", [3, 2.25, 3.5, 1.75])
def test_memory_increase_size_class(increase_factor):
    (x_1, y_1, t_1), _ = gen_data()

    memory_set = MemorySet(x_1, y_1, t_1, None)

    class_label = y_1[0]

    nb_initial_instances = memory_set.get_nb_instances_class(class_label=class_label)

    # for now the number of instance should be the same as the number of samples
    assert nb_initial_instances == memory_set.get_nb_samples_class(class_label=class_label)

    memory_set.increase_size_class(increase_factor, class_label=class_label)

    new_nb_instances = memory_set.get_nb_instances_class(class_label=class_label)

    assert new_nb_instances == int(round(nb_initial_instances * increase_factor))
    # the number of instance has been multiplied, not the number of samples
    assert new_nb_instances == int(round(memory_set.get_nb_samples_class(class_label=class_label) * increase_factor))


def test_memory_get_nb_samples_class():
    (x_1, y_1, t_1), _ = gen_data()

    memory_set = MemorySet(x_1, y_1, t_1, None)

    classes = memory_set.get_classes()

    for i, _class in enumerate(classes):
        nb_samples = len(np.where(y_1 == _class)[0])
        assert nb_samples == memory_set.get_nb_samples_class(_class)


def test_memory_classes_balance():
    (x_1, y_1, t_1), _ = gen_data()

    memory_set = MemorySet(x_1, y_1, t_1, None)

    memory_set.balance_classes()

    classes = memory_set.get_classes()

    nb_instances = 0
    for i, _class in enumerate(classes):
        if i != 0:
            assert nb_instances == memory_set.get_nb_instances_class(_class), \
                f"index {i}, class {_class} : {nb_instances} vs {memory_set.get_nb_instances_class(_class)}"
        nb_instances = memory_set.get_nb_instances_class(_class)
