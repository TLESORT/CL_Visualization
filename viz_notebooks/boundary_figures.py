

import torch
import torch.nn as nn
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

seed = 51
torch.manual_seed(51)
np.random.seed(51)


def plot_predictor_boundaries(predictor, figure_name):
    # X = torch.tensor(d.data[0]).view(-1, 512)

    dir_1 = predictor.weight[0] - predictor.weight[1]
    dir_1 /= torch.norm(dir_1)
    dir_2 = predictor.weight[3] - predictor.weight[2]
    dir_2 /= torch.norm(dir_2)

    # X_2d_lin_x = torch.mv(X, dir_1)
    # X_2d_lin_y = torch.mv(X, dir_2)

    proj_mat = torch.stack((dir_1, dir_2))

    lims = 6

    xa, ya = np.mgrid[-lims:lims:1000j, -lims:lims:1000j]
    x_display = np.vstack((xa.flatten(), ya.flatten())).T

    with torch.no_grad():
        X_proj = torch.mm(torch.tensor(x_display, dtype=torch.float), proj_mat)
        preds_display = predictor(X_proj).argmax(dim=1)

    limit_n = 250
    # indices = np.argwhere(d.data[1] < 4)
    # np.random.shuffle(indices)
    # indices = indices[:limit_n]

    plt.figure(figsize=(10, 10))
    # plt.scatter(X_2d_lin_x[indices], X_2d_lin_y[indices], c=d.data[1][indices], cmap='tab20')
    # plt.scatter(X_2d_lin_x[indices], X_2d_lin_y[indices], cmap='tab20')

    cs = plt.contourf(xa, ya, preds_display.reshape(1000, 1000), cmap='tab20', alpha=.2)

    plt.xlim(-lims, lims)
    plt.ylim(-lims, lims)

    plt.savefig(figure_name)
    plt.clf()
    plt.close()


with torch.no_grad():
    predictor = nn.Linear(2, 5, bias=True)
    plot_predictor_boundaries(predictor, figure_name="Linear.png")
    print("BIAS")
    print(predictor.bias)
    predictor.bias.fill_(0)
    plot_predictor_boundaries(predictor, figure_name="Linear_wo_bias.png")

    weight = predictor.weight.data
    weight = weight / torch.norm(weight, dim=1, keepdim=True)
    predictor.weight.data = weight
    plot_predictor_boundaries(predictor, figure_name="Linear_weight_norm.png")
