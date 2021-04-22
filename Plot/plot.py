
import matplotlib

matplotlib.use('agg')

import os
import pickle
from itertools import cycle
import numpy as np
import imageio
import argparse

#import matplotlib.animation as animation
#writer = animation.FFMpegFileWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)

import matplotlib.pyplot as plt
import sys
sys.path.append("..")
print(sys.path)

from Plot.plot_logs import plot_accuracies, \
    plot_angles_latent_output, \
    plot_accuracies_per_classes,\
    plot_loss,\
    plot_grad,\
    plot_grad_gif,\
    plot_Fisher,\
    plot_tsne_classes,\
    plot_tsne_tasks,\
    plot_weights_diff,\
    plot_mean_weights_dist, \
    plot_orthogonal_output_layers, \
    plot_norm_bias_output_layers

from Plot.comparative_plots import plot_comparative_accuracies, \
    plot_comparative_tsne_tasks, \
    plot_comparative_accuracies_per_classes, \
    plot_comparative_loss, \
    plot_comparative_accuracies_head


class Continual_Plot(object):
    """ this class gives function to plot continual algorithms evaluation and metrics"""

    def __init__(self, args):
        self.OutLayer = args.OutLayer
        self.log_dir = os.path.join(args.root_dir, "Logs", args.scenario_name)
        self.Fig_dir = os.path.join(args.root_dir, "Figures", args.scenario_name)

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        if not os.path.exists(self.Fig_dir):
            os.makedirs(self.Fig_dir)

        fast = True


    def plot_figures(self, method, log_dir=None):

        if log_dir is None:
            log_dir=self.log_dir

        # fast
        plot_accuracies(log_dir, self.Fig_dir, method)
        plot_accuracies_per_classes(log_dir, self.Fig_dir, method)

        # not fast
        plot_orthogonal_output_layers(log_dir, self.Fig_dir, method)
        plot_tsne_tasks(log_dir, self.Fig_dir, method)
        plot_tsne_classes(log_dir, self.Fig_dir, method)
        plot_mean_weights_dist(log_dir, self.Fig_dir, method)
        plot_Fisher(log_dir, self.Fig_dir, method)
        plot_loss(log_dir, self.Fig_dir, method)

        if not self.OutLayer=="SLDA":
            plot_grad(log_dir, self.Fig_dir, method)
            plot_weights_diff(log_dir, self.Fig_dir, method)
            plot_angles_latent_output(log_dir, self.Fig_dir, method)
            plot_norm_bias_output_layers(log_dir, self.Fig_dir, method)






        # plot_grad_gif(log_dir, Fig_dir, fast)


    def plot_comparison(self, list_methods, seed_list, head_list):

        #plot_comparative_accuracies(self.log_dir, self.Fig_dir, list_methods, seed_list)
        #plot_comparative_accuracies_per_classes(self.log_dir, self.Fig_dir, list_methods, seed_list)
        #plot_comparative_loss(self.log_dir, self.Fig_dir, list_methods, seed_list)
        plot_comparative_accuracies_head(self.log_dir, self.Fig_dir, "baseline", head_list, seed_list)

        new_log_dir = self.log_dir.replace("Logs", "seed-0/Logs")
        #plot_comparative_tsne_tasks(new_log_dir, self.Fig_dir, list_methods)
        print("in progress")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario_name', default="Disjoint", type=str, choices=["Disjoint","Domain"],
                        help='continual scenario')
    parser.add_argument('--root_dir', default="./Archives", type=str,
                        help='data directory name')
    parser.add_argument('--test_label', action='store_true', default=False,
                        help='define if we use task label at test')
    parser.add_argument('--dataset', default="MNIST", type=str, choices=["MNIST","mnist_fellowship","CIFAR10"],
                        help='dataset name')
    parser.add_argument('--pretrained_on', default="None", type=str,
                        choices=[None, "CIFAR10", "CIFAR100", "ImageNet"],
                        help='dataset source of a pretrained model')
    parser.add_argument('--num_tasks', type=int, default=5, help='Task number')
    parser.add_argument('--OutLayer', default="Linear", type=str,
                    choices=['Linear', 'CosLayer', 'SLDA'],
                    help='type of ouput layer used for the NN')

    args = parser.parse_args()
    args.root_dir = os.path.join(args.root_dir, args.dataset)
    if args.test_label:
        args.root_dir = os.path.join(args.root_dir, "MultiH")
    else:
        args.root_dir = os.path.join(args.root_dir, "SingleH")

    method_list = ["baseline", "ewc_diag", "rehearsal", "ewc_kfac", "ewc_diag_id","ogd"]
    method_list = ["baseline", "ewc_diag", "rehearsal", "ewc_kfac","ogd"]
    method_list = ["ewc_diag", "rehearsal", "ewc_kfac"]
    #method_list = ["ewc_kfac"]
    seed_list = [0,2,3,4,5,6,7]
    seed_list = [0,1]
    #method_list = ["ewc_diag", "rehearsal", "ewc_kfac"]
    #method_list = ["rehearsal"]

    head_list = ["Linear", "Linear_no_bias", "CosLayer", "SLDA", "MeanLayer", 'MIMO_Linear', "MIMO_CosLayer", "MIMO_Linear_no_bias",
                 "Linear_Masked", "Linear_no_bias_Masked", "CosLayer_Masked", 'MIMO_Linear_Masked', "MIMO_CosLayer_Masked", "MIMO_Linear_no_bias_Masked"]


    single_plot_seed=0
    plot_object = Continual_Plot(args)
    # for method in method_list:
    #     plot_object.plot_figures(method, log_dir=plot_object.log_dir.replace("Logs",f"seed-{single_plot_seed}/Logs"))
    plot_object.plot_comparison(method_list, seed_list, head_list)