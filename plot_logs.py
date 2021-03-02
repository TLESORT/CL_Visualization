import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import imageio
from matplotlib import colors
from copy import copy

from plot_utils import animate_heat_map
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sn




def flatten_results(results, type=""):
    nb_iterations = 0
    ind_task_transition = []
    for ind_task in range(len(results)):
        nb_iterations += len(results[ind_task])
        ind_task_transition.append(nb_iterations)

    if type == "loss" or type == "dist":
        shape_data = [nb_iterations]
    else:
        shape_data = [nb_iterations] + list(results[0][0].shape)
    np_flat_data = np.zeros(shape_data)

    iteration = 0
    for ind_task in range(len(results)):
        for i in range(len(results[ind_task])):
            np_flat_data[iteration] = np.array(results[ind_task][i])
            iteration += 1
    return np_flat_data, np.array(ind_task_transition)


def plot_Fisher(log_dir, Fig_dir, algo_name):
    print(f"Plot Fisher {algo_name}")

    file_name = os.path.join(log_dir, "{}_weights.pkl".format(algo_name))
    list_loss = None
    with open(file_name, 'rb') as fp:
        list_weight = pickle.load(fp)

    file_name = os.path.join(log_dir, "{}_Fishers.pkl".format(algo_name))
    list_Fisher = None
    with open(file_name, 'rb') as fp:
        list_Fisher = pickle.load(fp)

    nb_classes = list_weight[0][0][0].shape[0]

    fig, axs = plt.subplots(len(list_Fisher), 2)

    axs[0, 0].set_title('Weights')
    axs[0, 1].set_title('Fisher Information')

    for i in range(len(list_Fisher)):

        if i == 0:
            # pour la première fisher on prent les poids à l'initialization
            w = list_weight[0][0][0]
            b = list_weight[0][0][1].reshape(-1, 1)
        else:
            # sinon on prend les poids à la fin de la tâche précédentes
            w = list_weight[i - 1][-1][0]
            b = list_weight[i - 1][-1][1].reshape(-1, 1)

        layer = np.concatenate((w, b), axis=1).astype(np.float)

        fischer_w = np.array(list_Fisher[i])[:-nb_classes].reshape(nb_classes, 50)
        fischer_b = np.array(list_Fisher[i])[-nb_classes:].reshape(nb_classes, 1)
        fisher = np.concatenate((fischer_w, fischer_b), axis=1)

        #  linearly map the colors in the colormap from data values vmin to vmax
        axs[i, 0].imshow(layer, vmin=-0., vmax=1., cmap='PuBu_r')
        axs[i, 1].imshow(fisher, vmin=-0., vmax=1., cmap='PuBu_r')

        axs[i, 0].set_yticks([])
        axs[i, 0].get_xaxis().set_visible(False)
        axs[i, 1].get_yaxis().set_visible(False)
        axs[i, 1].get_xaxis().set_visible(False)
        axs[i, 0].set(ylabel='Task {}'.format(i))

    save_name = os.path.join(Fig_dir, f"{algo_name}_Fishers.png")
    plt.savefig(save_name)
    plt.clf()


def plot_mean_weights_dist(log_dir, Fig_dir, algo_name):
    print(f"Mean Weight diff {algo_name}")

    file_name = os.path.join(log_dir, "{}_dist.pkl".format(algo_name))

    list_dist = None
    with open(file_name, 'rb') as fp:
        list_dist = pickle.load(fp)

    np_dist, ind_task_transition = flatten_results(list_dist, type="dist")

    # we remove dist before start of training
    np_dist = np_dist[ind_task_transition[0]:-1]

    np_grad_reshaped = np_dist.reshape(np_dist.shape[0], -1)

    mean = np_grad_reshaped.mean(1)
    std = np_grad_reshaped.std(1)

    plt.plot(range(np_dist.shape[0]), mean, label="Distance")
    plt.fill_between(range(np_dist.shape[0]), mean - std, mean + std, alpha=0.4)

    # remove first transition which is start of training and remove offset
    xcoords = ind_task_transition[1:-1] - ind_task_transition[0]
    for i, xc in enumerate(xcoords):
        plt.axvline(x=xc, color='#000000', linewidth=0.1, linestyle='-.')

    plt.title("L2 distance between current model and model at the beginning of the task")
    plt.savefig(os.path.join(Fig_dir, "{}_Dist.png").format(algo_name))
    plt.clf()


def plot_orthogonal_output_layers(log_dir, Fig_dir, algo_name):
    file_name = os.path.join(log_dir, "{}_weights.pkl".format(algo_name))
    list_weight = None
    with open(file_name, 'rb') as fp:
        list_weight = pickle.load(fp)

    nb_classes = list_weight[0][0][0].shape[0]

    fig, axs = plt.subplots(1, len(list_weight), figsize=(15,3))

    for i in range(len(list_weight)):

        dot_products = np.zeros((nb_classes, nb_classes))

        for j in range(nb_classes):
            for k in range(nb_classes):
                w_j = list_weight[i][-1][0][j, :]
                w_k = list_weight[i][-1][0][k, :]
                assert len(w_j) == 50  # latent space

                dot_products[j][k] = w_j.dot(w_k)

                axs[i].imshow(dot_products, vmin=-0., vmax=1., cmap='PuBu_r')
                axs[i].get_xaxis().set_visible(False)
                axs[i].axis('off')
                if i == 0:
                    axs[i].set_title('Init')
                else:
                    axs[i].set_title('End Task {}'.format(i))

    save_name = os.path.join(Fig_dir, f"{algo_name}_Output_Layer_Correlation.png")
    plt.savefig(save_name)
    plt.clf()


def plot_weights_diff(log_dir, Fig_dir, algo_name):
    print(f"Weight diff {algo_name}")

    file_name = os.path.join(log_dir, "{}_weights.pkl".format(algo_name))
    list_loss = None
    with open(file_name, 'rb') as fp:
        list_weight = pickle.load(fp)

    fig, axs = plt.subplots(len(list_weight) + 1, 2)

    axs[0, 0].set_title('Weights')
    axs[0, 1].set_title('Weights Difference Since Last Task')
    previous_layer = None
    for i in range(len(list_weight)):

        if i == 0:
            # pour la première fisher on prent les poids à l'initialization
            w = list_weight[0][0][0]
            b = list_weight[0][0][1].reshape(-1, 1)
            previous_layer = np.concatenate((w, b), axis=1).astype(np.float)

            axs[i, 0].imshow(previous_layer, vmin=-0., vmax=1., cmap='PuBu_r')

            axs[i, 0].set_yticks([])
            axs[i, 0].get_xaxis().set_visible(False)
            axs[i, 1].axis('off')
            axs[i, 0].set(ylabel='Init'.format(i))

        # sinon on prend les poids à la fin de la tâche précédentes
        w = list_weight[i][-1][0]
        b = list_weight[i][-1][1].reshape(-1, 1)

        layer = np.concatenate((w, b), axis=1).astype(np.float)

        axs[i + 1, 0].imshow(layer, vmin=-0., vmax=1., cmap='PuBu_r')
        axs[i + 1, 1].imshow(layer - previous_layer, vmin=-0., vmax=1., cmap='PuBu_r')

        axs[i + 1, 0].set_yticks([])
        axs[i + 1, 0].get_xaxis().set_visible(False)
        axs[i + 1, 1].get_yaxis().set_visible(False)
        axs[i + 1, 1].get_xaxis().set_visible(False)
        axs[i + 1, 0].set(ylabel=f'Task {i}')

    save_name = os.path.join(Fig_dir, f"{algo_name}_Weight_Diff.png")
    plt.savefig(save_name)
    plt.clf()


def plot_tsne(log_dir, Fig_dir, algo_name):
    print(f"T-SNE {algo_name}")
    file_name = os.path.join(log_dir, "{}_Latent.pkl".format(algo_name))
    list_latent = None
    with open(file_name, 'rb') as fp:
        list_latent = pickle.load(fp)

    nb_tasks = len(list_latent)

    data = None
    label = None
    tsne_df = None

    for ind_task, (data, label) in enumerate(list_latent):
        model = TSNE(n_components=2, random_state=0)
        # the number of components = 2
        # default perplexity = 30
        # default learning rate = 200
        # default Maximum number of iterations for the optimization = 1000

        tsne_data = model.fit_transform(data)
        task_id = np.ones(label.shape[0]) * ind_task
        tsne_data = np.vstack((tsne_data.T, label, task_id)).T
        if ind_task == 0:
            tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label", "task"))
        else:
            tsne_df = pd.concat([tsne_df, pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label", "task"))])

    sn.FacetGrid(tsne_df, hue="label", height=6, col="task").map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
    plt.savefig(os.path.join(Fig_dir, "{}_tsne.png").format(algo_name))
    plt.clf()


def plot_loss(log_dir, Fig_dir, algo_name):
    print(f"Plot Loss {algo_name}")

    file_name = os.path.join(log_dir, "{}_loss.pkl".format(algo_name))
    list_loss = None
    with open(file_name, 'rb') as fp:
        list_loss = pickle.load(fp)

    flat_loss, ind_task_transition = flatten_results(list_loss, type="loss")

    plt.plot(range(flat_loss.shape[0]), flat_loss, label="Loss")

    xcoords = ind_task_transition

    for i, xc in enumerate(xcoords):
        if i==0:
            plt.axvline(x=xc, color='red', linewidth=0.5, linestyle='-')
        else:
            plt.axvline(x=xc, color='#000000', linewidth=0.1, linestyle='-.')

    plt.savefig(os.path.join(Fig_dir, "{}_Loss.png").format(algo_name))
    plt.clf()

def plot_accuracies(log_dir, Fig_dir, algo_name):
    print(f"Plot Accuracies {algo_name}")

    file_name = os.path.join(log_dir, "{}_accuracies.pkl".format(algo_name))
    dict_accuracies = None
    with open(file_name, 'rb') as fp:
        dict_accuracies = pickle.load(fp)

    flat_acc, ind_task_transition = flatten_results(dict_accuracies, type="acc")

    nb_correct_tr = flat_acc[:,0]
    nb_instances_tr = flat_acc[:,1]
    nb_correct_te = flat_acc[:,2]
    nb_instances_te = flat_acc[:,3]

    flat_acc_tr = np.divide(nb_correct_tr,nb_instances_tr)
    flat_acc_te = np.divide(nb_correct_te,nb_instances_te)

    plt.plot(range(flat_acc_tr.shape[0]), flat_acc_tr, label="Train Accuracy")
    plt.plot(range(flat_acc_te.shape[0]), flat_acc_te, label="Test Accuracy")

    xcoords = ind_task_transition
    for xc in xcoords:
        plt.axvline(x=xc, color='#000000', linewidth=0.1, linestyle='-.')

    plt.savefig(os.path.join(Fig_dir, "{}_Accuracy.png").format(algo_name))
    plt.clf()

def plot_accuracies_per_classes(log_dir, Fig_dir, algo_name):
    file_name = os.path.join(log_dir, "{}_accuracies_per_class.pkl".format(algo_name))
    dict_accuracies = None
    with open(file_name, 'rb') as fp:
        dict_accuracies = pickle.load(fp)

    flat_acc, ind_task_transition = flatten_results(dict_accuracies, type="acc")

    flat_te = flat_acc[:,1]
    flat_correct_te = flat_te[:,0]
    flat_wrong_te = flat_te[:,1]
    flat_nb_te = flat_te[:,2]

    fig, axs = plt.subplots(1, len(ind_task_transition), figsize=(15,3))

    # accuracy per class on test set (last epoch per task)
    for i, ind_epoch in enumerate(ind_task_transition):
        accuracy = np.divide(flat_correct_te[ind_epoch-1], flat_nb_te[ind_epoch-1]) * 100
        axs[i].bar(np.arange(len(accuracy))+1, accuracy, width = 0.8, tick_label = range(len(accuracy)))
        axs[i].set_xlim(0, len(accuracy)+1) # +2 for space management
        axs[i].set_box_aspect(1)
        if i == 0:
            axs[0].set_ylabel('Accuracy Per Class')
            axs[0].set_xlabel('Before Training')
        else:
            axs[i].set_xlabel(f'Task {i}')

    save_name = os.path.join(Fig_dir, f"{algo_name}_accuracies_per_class.png")
    plt.title('Accuracy at the end of each task')
    plt.savefig(save_name)
    plt.clf()



# grad of the last layer (without bias)
def plot_grad(log_dir, Fig_dir, algo_name):
    print(f"Plot Grad {algo_name}")

    file_name = os.path.join(log_dir, "{}_grad.pkl".format(algo_name))

    list_grad = None
    with open(file_name, 'rb') as fp:
        list_grad = pickle.load(fp)

    np_grad, ind_task_transition = flatten_results(list_grad, type="grad")

    np_grad_reshaped = np_grad.reshape(np_grad.shape[0], -1)

    # remove log before start of training
    np_grad_reshaped=np_grad_reshaped[ind_task_transition[1]:]

    # remove first ind and correct offset
    ind_task_transition = ind_task_transition[1:]-ind_task_transition[0]

    mean = np_grad_reshaped.mean(1)
    std = np_grad_reshaped.std(1)

    plt.plot(range(np_grad_reshaped.shape[0]), mean, label="Grad")
    plt.fill_between(range(np_grad_reshaped.shape[0]), mean - std, mean + std, alpha=0.4)

    xcoords = ind_task_transition
    for xc in xcoords:
        plt.axvline(x=xc, color='#000000', linewidth=0.1, linestyle='-.')

    plt.savefig(os.path.join(Fig_dir, "{}_Grad.png").format(algo_name))
    plt.clf()


def plot_grad_gif(log_dir, Fig_dir, fast=True):
    file_name = os.path.join(log_dir, "grad.pkl")

    list_grad = None
    with open(file_name, 'rb') as fp:
        list_grad = pickle.load(fp)

    np_grad = flatten_results(list_grad, type="grad")

    np_grad = np_grad - np_grad.min()
    np_grad = np_grad / np_grad.max()
    np_grad = (np_grad * 255)  # .astype(np.uint8)

    file_name = os.path.join(Fig_dir, 'gradients.gif')
    if fast:
        imageio.mimwrite(file_name, list(np_grad))
    # else:
    #     animate_heat_map(np_grad, filename=file_name)

    file_name = os.path.join(log_dir, "weights.pkl")
    list_weights = None
    with open(file_name, 'rb') as fp:
        list_weights = pickle.load(fp)

    nb_iterations = 0
    for ind_task in range(len(list_weights)):
        nb_iterations += len(list_weights[ind_task])

    len_list = nb_iterations
    height = list_weights[0][0][0].shape[0]
    width = list_weights[0][0][0].shape[1]
    tot_width = width + 11

    np_weights = np.zeros((len_list, height, tot_width))

    print(np_weights.shape)

    iteration = 0
    for ind_task in range(len(list_grad)):
        for i in range(len(list_grad[ind_task])):
            np_weights[i, :, :width] = np.array(list_weights[ind_task][i][0])
            np_weights[iteration, :, -1] = np.array(list_weights[ind_task][i][1])
            iteration += 1

    file_name = os.path.join(Fig_dir, 'weights.gif')
    if fast:
        imageio.mimsave(file_name, list(np_weights))
    else:
        animate_heat_map(np_grad, np_weights, filename=file_name)
        # fig, (ax1, ax2) = plt.subplots(2)
        # ax1 = sns.heatmap(np_grad[1], vmin=0, vmax=1)
        # ax2 = sns.heatmap(np_grad[0], vmin=0, vmax=1)

        # print(np_grad[1,:,:].shape)
        # ax1 = sns.heatmap(np_grad[0], vmin=0, vmax=1)
        # ax2 = sns.heatmap(np_grad[0], vmin=0, vmax=1)
        # plt.savefig(os.path.join(Fig_dir, "test.png"))
        # plt.clf()

    #
    # #ax2.yaxis.tick_right()
    # #ax2.tick_params(rotation=0)
    # plt.savefig(os.path.join(Fig_dir, "test.png"))
    # plt.clf()
