import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import imageio
from matplotlib import colors
from copy import copy

from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns

from plot_utils import animate_heat_map, flatten_results, angle_between

def plot_comparative_accuracies(log_dir, Fig_dir, list_methods):
    print(f"Plot Comparative Accuracies")
    print(list_methods)

    for method in list_methods:
        file_name = os.path.join(log_dir, "{}_accuracies.pkl".format(method))
        dict_accuracies = None
        with open(file_name, 'rb') as fp:
            dict_accuracies = pickle.load(fp)

        flat_acc, ind_task_transition = flatten_results(dict_accuracies, type="acc")

        nb_correct_tr = flat_acc[:, 0]
        nb_instances_tr = flat_acc[:, 1]
        nb_correct_te = flat_acc[:, 2]
        nb_instances_te = flat_acc[:, 3]

        flat_acc_tr = np.divide(nb_correct_tr, nb_instances_tr)
        flat_acc_te = np.divide(nb_correct_te, nb_instances_te)

        #plt.plot(range(flat_acc_tr.shape[0]), flat_acc_tr, label="method")
        plt.plot(range(flat_acc_te.shape[0]), flat_acc_te, label=method)

    # remove first ind and remove offset
    ind_task_transition = ind_task_transition[1:] - ind_task_transition[0]
    for xc in ind_task_transition:
        plt.axvline(x=xc, color='#000000', linewidth=0.1, linestyle='-.')

    plt.title('Evolution of Whole Test Accuracy')
    plt.legend()
    plt.savefig(os.path.join(Fig_dir, "Comparative_Accuracy.png"))
    plt.clf()
    plt.close()

def plot_comparative_tsne_tasks(log_dir, Fig_dir, list_methods):
    print(f"Comparative T-SNE Last Tasks")
    print(list_methods)

    list_latent_method =[]

    for method in list_methods:

        file_name = os.path.join(log_dir, "{}_Latent.pkl".format(method))
        with open(file_name, 'rb') as fp:
            list_latent = pickle.load(fp)

        nb_tasks = len(list_latent)
        list_latent_method.append(list_latent[nb_tasks-1])

    data = None
    label = None
    tsne_df = None
    dict_data = {}
    dict_data["Dim_1"] = []
    dict_data["Dim_2"] = []
    dict_data["label"] = []
    dict_data["method"] = []

    for ind_method, (data, label, task_labels) in enumerate(list_latent_method):
        model = TSNE(n_components=2, random_state=0)
        # the number of components = 2
        # default perplexity = 30
        # default learning rate = 200
        # default Maximum number of iterations for the optimization = 1000

        tsne_data = model.fit_transform(data)
        method_id = [list_methods[ind_method]]*label.shape[0]

        dict_data["Dim_1"] += list(tsne_data[:, 0])
        dict_data["Dim_2"] += list(tsne_data[:, 1])
        dict_data["label"] += list(task_labels)
        dict_data["method"] += method_id

    tsne_df = pd.DataFrame(data=dict_data)
    g = sns.FacetGrid(tsne_df, col="method", hue="label")
    g.map(sns.scatterplot, "Dim_1", "Dim_2", alpha=.7)
    g.add_legend()

    plt.savefig(os.path.join(Fig_dir, "Comparative_tsne.png"))
    plt.clf()
    plt.close()