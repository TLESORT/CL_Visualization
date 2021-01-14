
import matplotlib.pyplot as plt

import seaborn as sns
import pandas as pd


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