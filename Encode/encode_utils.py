import os
import pickle
from continuum.datasets import InMemoryDataset
from continuum.scenarios import ContinualScenario

def encode(model, scenario):

    # we save feature in eval mode
    model.eval()

    list_features = []
    list_labels = []
    list_tasks_labels = []
    
    for taskset in scenario:
        for i, (x,y,t) in taskset:
            features = model.feature_extractor(x)
            list_features.append(features)
            list_labels.append(y)
            list_tasks_labels.append(t)

    # create new scenario with encoded data
    cl_dataset = InMemoryDataset(x, y, t)
    encoded_scenario = ContinualScenario(cl_dataset)
    return encoded_scenario

def load_encoded(file_name):
    with open(file_name, 'rb') as fp:
        data = pickle.load(fp)
    encoded_scenario = ContinualScenario(data)
    return encoded_scenario

def save_encoded_data(file_name, encoded_data):
    with open(file_name, 'wb') as f:
        pickle.dump(encoded_data, f, pickle.HIGHEST_PROTOCOL)
    pass

def encode_scenario(data_dir, model, scenario, dataset_name, force_encode=False, save=True):


    if dataset_name in ["MNIST", "Fashion-MNIST", "KMNIST", "MNIST-Fellowship"]:
        #nothing happen here
        encoded_scenario = scenario
    else:
        data_name = os.path.join(data_dir, f"encode_{dataset_name}_{scenario.nb_tasks}.pkl")
        if os.path.isfile(data_name) and not force_encode:
            encoded_scenario = load_encoded(data_name)
        else:
            encoded_scenario = encode(model, scenario, dataset_name)
            if save:
                save_encoded_data(encoded_scenario.cl_dataset)

    return encoded_scenario