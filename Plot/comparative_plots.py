import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import imageio
from matplotlib import colors
from copy import copy

from Plot.plot_utils import animate_heat_map
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

    fig, axs = plt.subplots(len(list_Fisher), 2)

    axs[0, 0].set_title('Weights')
    axs[0, 1].set_title('Fisher Matrix')

    list_proc_Fisher = []
    list_Proc_Weights = []

    # we create variable to normalize figures between 0 and 1
    max_f = -1 * np.inf
    min_f = np.inf
    max_w = -1 * np.inf
    min_w = np.inf

    for i in range(len(list_Fisher)):
        nb_classes = list_weight[i][0][0].shape[0]

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

        if fisher.max() > max_f: max_f = fisher.max()
        if fisher.min() < min_f: min_f = fisher.min()
        if layer.max() > max_w: max_w = layer.max()
        if layer.min() < min_w: min_w = layer.min()

        list_proc_Fisher.append(fisher)
        list_Proc_Weights.append(layer)

    for i, (fisher, layer) in enumerate(zip(list_proc_Fisher, list_Proc_Weights)):
        fisher = fisher.repeat(2, axis=0).repeat(2, axis=1)  # grow image
        layer = layer.repeat(2, axis=0).repeat(2, axis=1)  # grow image

        #  linearly map the colors in the colormap from data values vmin to vmax
        pcm0 = axs[i, 0].imshow(layer, vmin=min_w, vmax=max_w, cmap='PuBu_r', interpolation='nearest')
        fig.colorbar(pcm0, ax=axs[i, 0], extend='max', orientation='vertical')
        pcm1 = axs[i, 1].imshow(fisher, vmin=-min_f, vmax=max_f, cmap='PuBu_r', interpolation='nearest')
        clb1 = fig.colorbar(pcm1, ax=axs[i, 1], extend='max')

        axs[i, 0].set_yticks([])
        axs[i, 0].get_xaxis().set_visible(False)
        axs[i, 1].get_yaxis().set_visible(False)
        axs[i, 1].get_xaxis().set_visible(False)
        axs[i, 0].set(ylabel='Task {}'.format(i))

    save_name = os.path.join(Fig_dir, f"{algo_name}_Fishers.png")
    plt.savefig(save_name)
    plt.clf()
    plt.close()


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
    plt.close()


from plot_utils import angle_between


def plot_orthogonal_output_layers(log_dir, Fig_dir, algo_name):
    file_name = os.path.join(log_dir, "{}_weights.pkl".format(algo_name))
    list_weight = None
    with open(file_name, 'rb') as fp:
        list_weight = pickle.load(fp)

    nb_classes = list_weight[0][0][0].shape[0]

    fig, axs = plt.subplots(1, len(list_weight), figsize=(15, 3))

    max_value = -1 * np.inf
    min_value = np.inf

    for i in range(len(list_weight)):

        angles_mat = np.zeros((nb_classes, nb_classes))

        for j in range(nb_classes):
            for k in range(nb_classes):
                w_j = list_weight[i][-1][0][j, :]
                w_k = list_weight[i][-1][0][k, :]
                assert len(w_j) == 50  # latent space
                assert len(w_k) == 50  # latent space
                angles_mat[j][k] = angle_between(w_j, w_k)

        pcm1 = axs[i].imshow(angles_mat, vmin=0, vmax=np.pi, cmap='PuBu_r')
        clb1 = fig.colorbar(pcm1, ax=axs[i], extend='max')
        axs[i].get_xaxis().set_visible(False)
        axs[i].axis('off')
        if i == 0:
            axs[i].set_title('Init')
        else:
            axs[i].set_title('End Task {}'.format(i))

    save_name = os.path.join(Fig_dir, f"{algo_name}_Output_Layer_Correlation.png")
    plt.title("Angles between output layer dimensions")
    plt.savefig(save_name)
    plt.clf()
    plt.close()


from numpy import linalg as LA


def plot_norm_bias_output_layers(log_dir, Fig_dir, algo_name):
    file_name = os.path.join(log_dir, "{}_weights.pkl".format(algo_name))
    list_weight = None
    with open(file_name, 'rb') as fp:
        list_weight = pickle.load(fp)

    nb_classes = list_weight[0][0][0].shape[0]

    fig, axs = plt.subplots(2, len(list_weight), figsize=(15, 6))

    for i in range(len(list_weight)):

        norm_mat = np.zeros(nb_classes)

        for j in range(nb_classes):
            norm_mat[j] = LA.norm(list_weight[i][-1][0][j, :])

        bias = list_weight[i][-1][1]

        assert bias.shape[0] == norm_mat.shape[0]

        axs[0, i].bar(np.arange(nb_classes) + 1, norm_mat, width=0.8, tick_label=range(nb_classes))
        axs[1, i].bar(np.arange(nb_classes) + 1, bias, width=0.8, tick_label=range(nb_classes))
        axs[0, i].set_xlim(0, nb_classes + 1)  # +1 for space management
        axs[1, i].set_xlim(0, nb_classes + 1)  # +1 for space management
        axs[0, i].set_box_aspect(1)
        axs[1, i].set_box_aspect(1)
        if i == 0:
            axs[0, 0].set_ylabel('Norm')
            axs[1, 0].set_ylabel('Bias')
            axs[1, 0].set_xlabel('Before Training')
        else:
            axs[1, i].set_xlabel(f'Task {i}')

    save_name = os.path.join(Fig_dir, f"{algo_name}_Norm_Bias_Output_Layer.png")
    plt.savefig(save_name)
    plt.clf()
    plt.close()


def plot_angles_latent_output(log_dir, Fig_dir, algo_name):
    print(f"Angles Latent Output {algo_name}")
    file_name = os.path.join(log_dir, "{}_Latent.pkl".format(algo_name))
    list_latent = None
    with open(file_name, 'rb') as fp:
        list_latent = pickle.load(fp)

    file_name = os.path.join(log_dir, "{}_weights.pkl".format(algo_name))
    list_weight = None
    with open(file_name, 'rb') as fp:
        list_weight = pickle.load(fp)

    nb_classes = list_weight[0][0][0].shape[0]

    fig, axs = plt.subplots(1, len(list_weight), figsize=(15, 3))

    for ind_task, (datas, labels, task_labels) in enumerate(list_latent):

        angles_mat = np.zeros(nb_classes)
        nb_instance_classes = np.zeros(nb_classes)

        for j in range(nb_classes):
            indexes_class = np.where(labels == j)[0]
            nb_instance_classes[j] = len(indexes_class)

        for i, (data, label, task_label) in enumerate(zip(datas, labels, task_labels)):
            label = int(label)
            weight_vector = list_weight[ind_task][-1][0][label, :]

            assert data.shape[0] == weight_vector.shape[0]
            angle = angle_between(data, weight_vector)
            angles_mat[label] = angle

        norm_angles_mat = np.divide(angles_mat, nb_instance_classes)
        axs[ind_task].bar(np.arange(nb_classes) + 1, norm_angles_mat, width=0.8, tick_label=range(nb_classes))
        axs[ind_task].set_xlim(0, nb_classes + 1)  # +1 for space management
        axs[ind_task].set_box_aspect(1)
        if ind_task == 0:
            axs[0].set_ylabel('Mean Angles')
            axs[0].set_xlabel('Before Training')
        else:
            axs[ind_task].set_xlabel(f'Task {ind_task}')

    save_name = os.path.join(Fig_dir, f"{algo_name}_angles_latent_output.png")
    plt.title('Angles Between Latent Vector and Output Layer')
    plt.savefig(save_name)
    plt.clf()
    plt.close()


def plot_weights_diff(log_dir, Fig_dir, algo_name):
    print(f"Weight diff {algo_name}")

    file_name = os.path.join(log_dir, "{}_weights.pkl".format(algo_name))
    list_loss = None
    with open(file_name, 'rb') as fp:
        list_weight = pickle.load(fp)

    fig, axs = plt.subplots(len(list_weight), 2)

    axs[0, 0].set_title('Weights')
    axs[0, 1].set_title('Weights Difference Since Last Task')
    previous_layer = None
    for i in range(len(list_weight)):

        # sinon on prend les poids à la fin de la tâche précédentes
        w = list_weight[i][-1][0]
        b = list_weight[i][-1][1].reshape(-1, 1)

        layer = np.concatenate((w, b), axis=1).astype(np.float)

        # pcm0 = axs[i, 0].imshow(layer, vmin=-0., vmax=1., cmap='PuBu_r')
        pcm0 = axs[i, 0].imshow(layer, cmap='PuBu_r')
        clb1 = fig.colorbar(pcm0, ax=axs[i, 0], extend='max')

        if i > 0:
            # pcm1 = axs[i, 1].imshow(layer - previous_layer, vmin=-0., vmax=1., cmap='PuBu_r')
            pcm1 = axs[i, 1].imshow(layer - previous_layer, cmap='PuBu_r')
            clb1 = fig.colorbar(pcm1, ax=axs[i, 1], extend='max')
            axs[i, 0].set(ylabel=f'Task {i}')
        else:
            axs[i, 1].axis('off')
            axs[i, 0].set(ylabel=f'Init')

        previous_layer = layer

        axs[i, 0].set_yticks([])
        axs[i, 0].get_xaxis().set_visible(False)
        axs[i, 1].get_yaxis().set_visible(False)
        axs[i, 1].get_xaxis().set_visible(False)

    save_name = os.path.join(Fig_dir, f"{algo_name}_Weight_Diff.png")
    plt.savefig(save_name)
    plt.clf()
    plt.close()


def plot_tsne_classes(log_dir, Fig_dir, algo_name):
    print(f"T-SNE Classes {algo_name}")
    file_name = os.path.join(log_dir, "{}_Latent.pkl".format(algo_name))
    list_latent = None
    with open(file_name, 'rb') as fp:
        list_latent = pickle.load(fp)

    nb_tasks = len(list_latent)

    data = None
    label = None
    tsne_df = None

    for ind_task, (data, label, task_labels) in enumerate(list_latent):
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
    plt.savefig(os.path.join(Fig_dir, "{}_tsne_classes.png").format(algo_name))
    plt.clf()
    plt.close()


def plot_tsne_tasks(log_dir, Fig_dir, algo_name):
    print(f"T-SNE Tasks {algo_name}")
    file_name = os.path.join(log_dir, "{}_Latent.pkl".format(algo_name))
    with open(file_name, 'rb') as fp:
        list_latent = pickle.load(fp)

    nb_tasks = len(list_latent)

    data = None
    label = None
    tsne_df = None

    for ind_task, (data, label, task_labels) in enumerate(list_latent):
        model = TSNE(n_components=2, random_state=0)
        # the number of components = 2
        # default perplexity = 30
        # default learning rate = 200
        # default Maximum number of iterations for the optimization = 1000

        tsne_data = model.fit_transform(data)
        task_id = np.ones(label.shape[0]) * ind_task
        # correct one to make work
        tsne_data = np.vstack((tsne_data.T, task_labels, task_id)).T
        if ind_task == 0:
            tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label", "task"))
        else:
            tsne_df = pd.concat([tsne_df, pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label", "task"))])

    sn.FacetGrid(tsne_df, hue="label", height=6, col="task").map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
    plt.savefig(os.path.join(Fig_dir, "{}_tsne_tasks.png").format(algo_name))
    plt.clf()
    plt.close()


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
        if i == 0:
            plt.axvline(x=xc, color='red', linewidth=0.5, linestyle='-')
        else:
            plt.axvline(x=xc, color='#000000', linewidth=0.1, linestyle='-.')

    plt.savefig(os.path.join(Fig_dir, "{}_Loss.png").format(algo_name))
    plt.clf()
    plt.close()


def plot_accuracies(log_dir, Fig_dir, algo_name):
    print(f"Plot Accuracies {algo_name}")

    file_name = os.path.join(log_dir, "{}_accuracies.pkl".format(algo_name))
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

    plt.plot(range(flat_acc_tr.shape[0]), flat_acc_tr, label="Train Accuracy")
    plt.plot(range(flat_acc_te.shape[0]), flat_acc_te, label="Test Accuracy")

    xcoords = ind_task_transition
    for xc in xcoords:
        plt.axvline(x=xc, color='#000000', linewidth=0.1, linestyle='-.')

    plt.title('Evolution of Accuracy')
    plt.savefig(os.path.join(Fig_dir, "{}_Accuracy.png").format(algo_name))
    plt.clf()
    plt.close()


def plot_accuracies_per_classes(log_dir, Fig_dir, algo_name):
    print(f"Plot Accuracies per Class {algo_name}")
    file_name = os.path.join(log_dir, "{}_accuracies_per_class.pkl".format(algo_name))
    dict_accuracies = None
    with open(file_name, 'rb') as fp:
        dict_accuracies = pickle.load(fp)

    flat_acc, ind_task_transition = flatten_results(dict_accuracies, type="acc")

    flat_te = flat_acc[:, 1]
    flat_correct_te = flat_te[:, 0]
    flat_wrong_te = flat_te[:, 1]
    flat_nb_te = flat_te[:, 2]

    fig, axs = plt.subplots(1, len(ind_task_transition), figsize=(15, 3))

    # accuracy per class on test set (last epoch per task)
    for i, ind_epoch in enumerate(ind_task_transition):
        accuracy = np.divide(flat_correct_te[ind_epoch - 1], flat_nb_te[ind_epoch - 1]) * 100
        axs[i].bar(np.arange(len(accuracy)) + 1, accuracy, width=0.8, tick_label=range(len(accuracy)))
        axs[i].set_xlim(0, len(accuracy) + 1)  # +2 for space management
        axs[i].set_box_aspect(1)
        if i == 0:
            axs[0].set_ylabel('Accuracy Per Class')
            axs[0].set_xlabel('Before Training')
        else:
            axs[i].set_xlabel(f'Task {i}')

    save_name = os.path.join(Fig_dir, f"{algo_name}_accuracies_per_class.png")
    plt.title('Accuracy per Class at the end of each task')
    plt.savefig(save_name)
    plt.clf()
    plt.close()


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
    np_grad_reshaped = np_grad_reshaped[ind_task_transition[0]:]

    # remove first ind and correct offset
    ind_task_transition = ind_task_transition[1:] - ind_task_transition[0]
    norm = LA.norm(np_grad_reshaped, axis=1)
    std = np_grad_reshaped.std(1)

    assert norm.shape[0] == np_grad_reshaped.shape[0]

    plt.plot(range(np_grad_reshaped.shape[0]), norm, label="Grad")
    # plt.fill_between(range(np_grad_reshaped.shape[0]), mean - std, mean + std, alpha=0.4)

    xcoords = ind_task_transition
    for xc in xcoords:
        plt.axvline(x=xc, color='#000000', linewidth=0.1, linestyle='-.')

    plt.savefig(os.path.join(Fig_dir, "{}_Grad.png").format(algo_name))
    plt.clf()
    plt.close()

