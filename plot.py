
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
        self.log_dir = os.path.join(args.Root_dir, "Logs", args.scenario_name)
        self.Fig_dir = os.path.join(args.Root_dir, "Figures", args.scenario_name)

        fast = True


    def plot_figures(self):

        name_list = ["baseline","ewc_diag", "rehearsal"]
        name_list = ["ewc_kfac"]
        for name in name_list:
            plot_orthogonal_output_layers(self.log_dir, self.Fig_dir, name)

            plot_tsne(self.log_dir, self.Fig_dir, name)
            plot_weights_diff(self.log_dir, self.Fig_dir, name)
            plot_mean_weights_dist(self.log_dir, self.Fig_dir, name)

            plot_Fisher(self.log_dir, self.Fig_dir, name)

            plot_loss(self.log_dir, self.Fig_dir, name)
            plot_grad(self.log_dir, self.Fig_dir, name)

            # plot_grad_gif(log_dir, Fig_dir, fast)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario_name', default="Disjoint", type=str,
                        help='continual scenario')
    parser.add_argument('--Root_dir', default="./Archives", type=str,
                        help='data directory name')

    args = parser.parse_args()

    Continual_Plot(args).plot_figures()