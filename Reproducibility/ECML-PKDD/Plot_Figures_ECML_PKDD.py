
import matplotlib
matplotlib.use('agg')

import os
import sys
sys.path.append("../..")

from Plot.comparative_plots import plot_comparative_accuracies, \
    plot_comparative_tsne_tasks



log_dir_multi_head = os.path.join("../../Archives", "mnist_fellowship", "MultiH", "Logs", "Disjoint")
log_dir_single_head = os.path.join("../../Archives", "mnist_fellowship", "SingleH", "Logs", "Disjoint")
fig_dir_multi_head = "MultiH"
fig_dir_single_head = "SingleH"


if not os.path.exists(fig_dir_multi_head):
    os.makedirs(fig_dir_multi_head)

if not os.path.exists(fig_dir_single_head):
    os.makedirs(fig_dir_single_head)

methods_list = ["baseline", "ewc_diag", "rehearsal", "ewc_kfac" ,"ogd"]
seed_list = [0,2,3,4,5,6,7]

plot_comparative_accuracies(log_dir_multi_head, fig_dir_multi_head, methods_list, seed_list)
plot_comparative_accuracies(log_dir_single_head, fig_dir_single_head, methods_list, seed_list)
plot_comparative_tsne_tasks(os.path.join(log_dir_single_head,"seed-0"), fig_dir_single_head, methods_list)
