import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import imageio
from matplotlib import colors
from copy import copy
from itertools import cycle

from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns

from Plot.plot_utils import animate_heat_map, flatten_results, angle_between

def plot_comparative_loss(log_dir, Fig_dir, list_methods, list_seed):
    print(f"Comparative Loss")
    print(list_methods)
    ind_task_transition=None
    for method in list_methods:
        style_c = cycle(['-', '--', ':', '-.'])
        list_results = []
        for seed in list_seed:
            seed_log_dir = log_dir.replace("Logs", f"seed-{seed}/Logs")
            file_name = os.path.join(seed_log_dir, "{}_loss.pkl".format(method))
            list_loss = None
            with open(file_name, 'rb') as fp:
                list_loss = pickle.load(fp)

            np_dist, ind_task_transition = flatten_results(list_loss, type="loss")

            # we remove dist before start of training
            np_dist = np_dist[ind_task_transition[0]:-1]
            list_results.append(np_dist)

        np_results = np.array(list_results)
        assert np_results.shape[0] == len(list_seed)

        mean = np.mean(np_results, axis=0)
        std = np.std(np_results, axis=0)
        size = np_results.shape[1]

        plt.plot(range(size), mean, label=method, linestyle=next(style_c))
        plt.fill_between(range(size), mean - std,
                         mean + std, alpha=0.4)

    # remove first transition which is start of training and remove offset
    xcoords = ind_task_transition[1:-1] - ind_task_transition[0]
    for i, xc in enumerate(xcoords):
        plt.axvline(x=xc, color='#000000', linewidth=0.1, linestyle='-.')

    plt.ylabel('Loss')
    plt.xlabel('Batch Id')
    plt.legend()
    plt.title("Evolution of Loss")
    plt.savefig(os.path.join(Fig_dir, "Comparative_Loss.png"))
    plt.clf()
    plt.close()

def plot_comparative_weights_diff(log_dir, Fig_dir, list_methods, list_seed):
    print(f"Comparative Weight dist")
    print(list_methods)
    ind_task_transition=None
    for method in list_methods:
        style_c = cycle(['-', '--', ':', '-.'])
        list_results = []
        for seed in list_seed:
            seed_log_dir = log_dir.replace("Logs", f"seed-{seed}/Logs")
            file_name = os.path.join(seed_log_dir, "{}_dist.pkl".format(method))
            list_dist = None
            with open(file_name, 'rb') as fp:
                list_dist = pickle.load(fp)

            np_dist, ind_task_transition = flatten_results(list_dist, type="dist")

            # we remove dist before start of training
            np_dist = np_dist[ind_task_transition[0]:-1]
            list_results.append(np_dist)

        np_results = np.array(list_results)
        assert np_results.shape[0] == len(list_seed)

        mean = np.mean(np_results, axis=0)
        std = np.std(np_results, axis=0)
        size = np_results.shape[1]

        plt.plot(range(size), mean, label=method, linestyle=next(style_c))
        plt.fill_between(range(size), mean - std,
                         mean + std, alpha=0.4)

    # remove first transition which is start of training and remove offset
    xcoords = ind_task_transition[1:-1] - ind_task_transition[0]
    for i, xc in enumerate(xcoords):
        plt.axvline(x=xc, color='#000000', linewidth=0.1, linestyle='-.')

    plt.ylabel('L2 distance')
    plt.xlabel('Batch Id')
    plt.legend()
    plt.title("L2 distance between current model and model at the beginning of the task")
    plt.savefig(os.path.join(Fig_dir, "Comparative_Dist.png"))
    plt.clf()
    plt.close()


def plot_comparative_accuracies(log_dir, Fig_dir, list_methods, list_seed):
    print(f"Plot Comparative Accuracies")
    print(list_methods)

    baseline_ref_mean=None
    baseline_ref_std=None

    for method in list_methods:
        style_c = cycle(['-', '--', ':', '-.'])
        list_results = []
        for seed in list_seed:
            seed_log_dir = log_dir.replace("Logs", f"seed-{seed}/Logs")
            file_name = os.path.join(seed_log_dir,f"{method}_accuracies.pkl")
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

            list_results.append(flat_acc_te)

            #plt.plot(range(flat_acc_tr.shape[0]), flat_acc_tr, label="method")
        np_results = np.array(list_results)
        assert np_results.shape[0]==len(list_seed)

        mean = np.mean(np_results, axis=0)
        std = np.std(np_results, axis=0)
        size = np_results.shape[1]

        # if method=="baseline":
        #     baseline_ref_mean = mean
        #     baseline_ref_std = std
        # else:
        #     if size < baseline_ref_mean.shape[0]:
        #         print("Temporary fix for the first task logs should be removed asap")
        #
        #         new_mean = baseline_ref_mean
        #         new_std = baseline_ref_std
        #         print(baseline_ref_mean[-1])
        #         print(mean[-1])
        #         new_mean[6:] = mean[2:]
        #         new_std[6:] = std[2:]
        #         mean=new_mean
        #         std=new_std
        #         size=mean.shape[0]


        plt.plot(range(size), mean, label=method, linestyle=next(style_c))
        plt.fill_between(range(size), mean - std,
                         mean + std, alpha=0.4)

    # remove first ind and remove offset
    ind_task_transition = ind_task_transition[1:] - ind_task_transition[0]
    for xc in ind_task_transition:
        plt.axvline(x=xc, color='#000000', linewidth=0.1, linestyle='-.')

    #plt.title('Evolution of Whole Test Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.savefig(os.path.join(Fig_dir, "Comparative_Accuracy.png"))
    plt.clf()
    plt.close()

def plot_comparative_accuracies_per_classes(log_dir, Fig_dir, list_method, list_seed):
    print(f"Plot Comparative Accuracies per Class")

    num_method = len(list_method)
    fig, axs = plt.subplots(1, num_method, figsize=(num_method * 5,3))

    for i, method in enumerate(list_method):
        list_accuracy = []
        for seed in list_seed:
            seed_log_dir = log_dir.replace("Logs", f"seed-{seed}/Logs")
            file_name = os.path.join(seed_log_dir, "{}_accuracies_per_class.pkl".format(method))
            dict_accuracies = None
            with open(file_name, 'rb') as fp:
                dict_accuracies = pickle.load(fp)

            flat_acc, ind_task_transition = flatten_results(dict_accuracies, type="acc")

            # we take final results only
            # flat_te -> [epoch_id, 1, classes]
            flat_te = flat_acc[:, 1]
            flat_correct_te = flat_te[-1, 0]
            flat_wrong_te = flat_te[-1, 1]
            flat_nb_te = flat_te[-1, 2]

            accuracy = np.divide(flat_correct_te, flat_nb_te) * 100
            list_accuracy.append(accuracy)

        np_accuracy = np.array(list_accuracy)
        mean = np.mean(np_accuracy, axis=0)
        std = np.std(np_accuracy, axis=0)

        axs[i].bar(np.arange(len(mean)) + 1, mean, yerr=std, width=0.8, tick_label=range(len(accuracy)))
        axs[i].set_xlim(0, len(mean) + 1)  # +2 for space management
        #axs[i].set_box_aspect(1)
        axs[i].set_xlabel(method)

    save_name = os.path.join(Fig_dir, f"comparative_accuracies_per_class.png")
    #plt.title('Accuracy per Class at the end of each task')
    plt.tight_layout()
    plt.savefig(save_name)
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


