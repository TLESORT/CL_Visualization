
import matplotlib.pyplot as plt

import seaborn as sns
import pandas as pd
import numpy as np


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def animate_heat_map(data_grad, data_weights, filename):
    fig, (ax1, ax2) = plt.subplots(2)
    df1 = pd.DataFrame(data_weights[0])
    df2 = pd.DataFrame(data_grad[0])

    fig, (ax1, ax2) = plt.subplots(2)

    sns.heatmap(df1, cmap="icefire", ax=ax1, cbar=False)
    fig.colorbar(ax1.collections[0], ax=ax1, location="top", use_gridspec=False, pad=0.2)
    ax1.set_title('Weights')
    ax1.xaxis.set_visible(False)

    sns.heatmap(df2, cmap="icefire", ax=ax2, cbar=False)
    fig.colorbar(ax2.collections[0], ax=ax2, location="bottom", use_gridspec=False, pad=0.2)
    ax2.set_title('Gradients')
    ax2.xaxis.set_visible(False)

    def init():
        plt.clf()
        df1 = pd.DataFrame(data_weights[0])
        df2 = pd.DataFrame(data_grad[0])

        fig, (ax1, ax2) = plt.subplots(2)

        sns.heatmap(df1, cmap="icefire", ax=ax1, cbar=False)
        fig.colorbar(ax1.collections[0], ax=ax1, location="top", use_gridspec=False, pad=0.2)
        ax1.set_title('Weights')
        ax1.xaxis.set_visible(False)

        sns.heatmap(df2, cmap="icefire", ax=ax2, cbar=False)
        fig.colorbar(ax2.collections[0], ax=ax2, location="bottom", use_gridspec=False, pad=0.2)
        ax2.set_title('Gradients')
        ax2.xaxis.set_visible(False)

    def animate(i):
        plt.clf()
        fig, (ax1, ax2) = plt.subplots(2)
        df1 = pd.DataFrame(data_weights[i])
        df2 = pd.DataFrame(data_grad[i])

        sns.heatmap(df1, cmap="icefire", ax=ax1, cbar=False)
        fig.colorbar(ax1.collections[0], ax=ax1, location="top", use_gridspec=False, pad=0.2)
        ax1.set_title('Weights')
        ax1.xaxis.set_visible(False)

        sns.heatmap(df2, cmap="icefire", ax=ax2, cbar=False)
        fig.colorbar(ax2.collections[0], ax=ax2, location="bottom", use_gridspec=False, pad=0.2)
        ax2.set_title('Gradients')
        ax2.xaxis.set_visible(False)

    # anim = animation.FuncAnimation(fig, animate, init_func=init, frames=data_grad.shape[0], interval=20)
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=10, interval=20)
    anim.save(filename, writer='imagemagick')


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
