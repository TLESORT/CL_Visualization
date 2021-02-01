import numpy as np
from memory import MemorySet
from torch.utils import data
import pytest

SIZE_MEMORY=20

def gen_data():
    x_1 = np.random.randint(0, 255, size=(SIZE_MEMORY, 32, 32, 3))
    y_1 = []
    for i in range(10):
        y_1.append(np.ones(2) * i)
    y_1 = np.concatenate(y_1)
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

    assert len(memory_set) == int(before_len * increase_factor)
    # the number of samples stay the same only the id list grows
    assert memory_set.get_nb_samples() == before_nb_samples

def test_concatenate():
    (x_1, y_1, t_1), (x_2, y_2, t_2) = gen_data()

    memory_set_1 = MemorySet(x_1, y_1, t_1, None)
    memory_set_2 = MemorySet(x_2, y_2, t_2, None)

    memory_set_1.concatenate(other_memory_set=memory_set_2)

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