
import matplotlib

matplotlib.use('agg')

import os
import pickle
from itertools import cycle
import numpy as np
import imageio
import argparse

import matplotlib.animation as animation

writer = animation.FFMpegFileWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)

import matplotlib.pyplot as plt

from plot_logs import plot_loss,\
    plot_grad,\
    plot_grad_gif,\
    plot_Fisher,\
    plot_tsne,\
    plot_weights_diff,\
    plot_mean_weights_dist, \
    plot_orthogonal_output_layers

class Continual_Plot(object):
    """ this class gives function to plot continual algorithms evaluation and metrics"""

    def __init__(self, args):
        self.log_dir = os.path.join(args.root_dir, "Logs", args.scenario_name)
        self.Fig_dir = os.path.join(args.root_dir, "Figures", args.scenario_name)

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        if not os.path.exists(self.Fig_dir):
            os.makedirs(self.Fig_dir)

        fast = True


    def plot_figures(self, method):
        plot_orthogonal_output_layers(self.log_dir, self.Fig_dir, method)

        plot_tsne(self.log_dir, self.Fig_dir, method)
        plot_weights_diff(self.log_dir, self.Fig_dir, method)
        plot_mean_weights_dist(self.log_dir, self.Fig_dir, method)

        plot_Fisher(self.log_dir, self.Fig_dir, method)

        plot_loss(self.log_dir, self.Fig_dir, method)
        plot_grad(self.log_dir, self.Fig_dir, method)

        # plot_grad_gif(log_dir, Fig_dir, fast)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario_name', default="Disjoint", type=str, choices=["Disjoint","Domain"],
                        help='continual scenario')
    parser.add_argument('--root_dir', default="./Archives", type=str,
                        help='data directory name')
    parser.add_argument('--dataset', default="MNIST", type=str, choices=["MNIST","mnist_fellowship"],
                        help='dataset name')

    args = parser.parse_args()

    args.root_dir = os.path.join(args.root_dir, args.dataset)
    plot_object = Continual_Plot(args)

    method_list = ["baseline", "ewc_diag", "rehearsal", "ewc_kfac"]
    method_list = ["ewc_diag", "rehearsal", "ewc_kfac"]
    for method in method_list:
        plot_object.plot_figures(method)