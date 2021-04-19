import os
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from continuum.datasets import InMemoryDataset
from continuum.scenarios import ContinualScenario

def encode(model, scenario):

    # we save feature in eval mode
    model.eval()

    list_features = []
    list_labels = []
    list_tasks_labels = []
    
    for taskset in scenario:
        loader = DataLoader(taskset, shuffle=True)
        i=0
        for x,y,t in loader:
            i+=1
            features = model.feature_extractor(x.cuda())
            list_features.append(features.detach().cpu())
            list_labels.append(y)
            list_tasks_labels.append(t)

    # convert into torch tensor
    feature_vector  = torch.cat(list_features).numpy()
    label_vector  = torch.cat(list_labels).numpy()
    tasks_labels_vector  = torch.cat(list_tasks_labels).numpy()

    # create new scenario with encoded data
    cl_dataset = InMemoryDataset(feature_vector, label_vector, tasks_labels_vector)
    encoded_scenario = ContinualScenario(cl_dataset)
    return encoded_scenario

def load_encoded(file_name):
    print("Load encoded data")
    with open(file_name, 'rb') as fp:
        data = pickle.load(fp)
    encoded_scenario = ContinualScenario(data)
    return encoded_scenario

def save_encoded_data(file_name, encoded_data):
    print("Save encoded data")
    with open(file_name, 'wb') as f:
        pickle.dump(encoded_data, f, pickle.HIGHEST_PROTOCOL)
    pass

def encode_scenario(data_dir, scenario, model, name, force_encode=False, save=True):

    data_path = os.path.join(data_dir, f"{name}.pkl")
    if os.path.isfile(data_path) and not force_encode:
        encoded_scenario = load_encoded(data_path)
    else:
        encoded_scenario = encode(model, scenario)
        if save:
            save_encoded_data(data_path, encoded_scenario.cl_dataset)

    return encoded_scenario