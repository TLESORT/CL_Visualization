import os
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from continuum.datasets import InMemoryDataset
from continuum.scenarios import ContinualScenario

def encode(model, scenario, batch_size, dataset, train):

    # we save feature in eval mode
    model.eval()

    list_features = []
    list_labels = []
    list_tasks_labels = []
    with torch.no_grad():
        for taskset in scenario:
            loader = DataLoader(taskset, shuffle=False, batch_size=batch_size)
            for i, (x,y,t) in enumerate(loader):
                if dataset in ["Core50", "Core10Lifelong"] and train:
                    assert batch_size >= 32
                    # divide fps by 8 to make dataset lighter (we can do it because there are a lot of redundant data)
                    # select 1/8 of samples
                    bs=len(y)
                    if bs>= 8:
                        nb_samples = int(bs/8)
                        indexes = torch.randperm(bs)[:nb_samples]
                        x=x[indexes]
                        y=y[indexes]
                        t=t[indexes]
                features = model.feature_extractor(x.cuda())
                list_features.append(features.detach().cpu())
                list_labels.append(y)
                list_tasks_labels.append(t)

    # convert into torch tensor
    feature_vector  = torch.cat(list_features).numpy()
    label_vector  = torch.cat(list_labels).numpy()
    tasks_labels_vector  = torch.cat(list_tasks_labels).numpy()

    # create new scenario with encoded data
    cl_dataset = InMemoryDataset(feature_vector, label_vector, tasks_labels_vector, data_type="tensor")
    encoded_scenario = ContinualScenario(cl_dataset)
    return encoded_scenario

def load_encoded(file_name):
    print("Load encoded data")
    with open(file_name, 'rb') as fp:
        cl_dataset = pickle.load(fp)
    encoded_scenario = ContinualScenario(cl_dataset)
    return encoded_scenario

def save_encoded_data(file_name, encoded_data):
    print("Save encoded data")
    with open(file_name, 'wb') as f:
        pickle.dump(encoded_data, f, pickle.HIGHEST_PROTOCOL)

def encode_scenario(data_dir, scenario, model, batch_size, name, force_encode=False, save=True, train=True, dataset=None):

    data_path = os.path.join(data_dir, f"{name}.pkl")
    if os.path.isfile(data_path) and not force_encode:
        encoded_scenario = load_encoded(data_path)
    else:
        print(f"Encoding {data_path}")
        encoded_scenario = encode(model, scenario, batch_size, dataset=dataset, train=train)
        if save:
            save_encoded_data(data_path, encoded_scenario.cl_dataset)

    return encoded_scenario