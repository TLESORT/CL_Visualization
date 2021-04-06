
import matplotlib

from torch.utils import data
from nngeometry.layercollection import LayerCollection
from nngeometry.metrics import FIM
from nngeometry.object import PMatDiag
matplotlib.use('agg')

import os
import torch
from torch.utils.data import DataLoader
import pickle
from itertools import cycle
import numpy as np
import imageio
import argparse

import matplotlib.animation as animation

writer = animation.FFMpegFileWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)

import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from utils import get_dataset, get_model
from Methods.trainer import Trainer

class Continual_Analysis(Trainer):
    """ this class gives function to plot continual algorithms evaluation and metrics"""

    def __init__(self, args):
        super().__init__(args)
        self.list_Fisher = []

    def analysis(self):


        for ind_task, task_set in enumerate(self.scenario_tr):
            data_loader_tr = DataLoader(task_set, batch_size=self.batch_size, shuffle=True, num_workers=6)
            self.model = self.load_model(ind_task)
            self.compute_Fishers(data_loader_tr, self.model, ind_task)
            self.log_latent(ind_task)


    def compute_last_layer_fisher(self, loader, model):

        layer_collection_all_layers = LayerCollection.from_model(model)
        layer_collection_last_layer = LayerCollection()
        layer_collection_last_layer.add_layer(*list(layer_collection_all_layers.layers.items())[-1])

        F_diag = FIM(layer_collection=layer_collection_last_layer,
                     model=model,
                     loader=loader,
                     representation=PMatDiag,
                     n_output=self.scenario_tr.nb_classes,
                     variant='classif_logits',
                     device='cuda')
        return F_diag, None

    def load_model(self, ind_task):

        model_weights_path = os.path.join(self.log_dir, f"Model_Task_{ind_task}.pth")
        pretrained_weights = torch.load(model_weights_path)
        self.model.load_state_dict(pretrained_weights)

    def compute_Fishers(self,loader, model, ind_task):
        F_diag, v0 = self.compute_last_layer_fisher(loader, model)
        self.list_Fisher.append(F_diag.get_diag().detach().cpu())

    def log_latent(self, model, ind_task):
        model.eval()

        nb_latent_vector = 200

        latent_vectors = np.zeros([0, 50])
        y_vectors = np.zeros([0])
        t_vectors = np.zeros([0])

        for i_, (x_, y_, t_) in enumerate(self.eval_tr_loader):

            # data does not fit to the model if size<=1
            if x_.size(0) <= 1:
                continue
            x_ = x_.cuda()
            latent_vector = model(x_, latent_vector=True).detach().cpu()
            latent_vectors = np.concatenate([latent_vectors, latent_vector], axis=0)
            y_vectors = np.concatenate([y_vectors, np.array(y_)], axis=0)
            t_vectors = np.concatenate([t_vectors, np.array(t_)], axis=0)

            if len(y_vectors) >= nb_latent_vector:
                break
        latent_vectors = latent_vectors[:nb_latent_vector]
        y_vectors = y_vectors[:nb_latent_vector]
        t_vectors = t_vectors[:nb_latent_vector]
        self.list_latent.append([latent_vectors, y_vectors, t_vectors])

    def save_latent(self, list_methods, seed_list):
        file_name = os.path.join(self.log_dir, f"{self.name_algo}_Latent.pkl")
        with open(file_name, 'wb') as f:
            pickle.dump(self.list_latent, f, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario_name', default="Disjoint", type=str, choices=["Disjoint","Domain"],
                        help='continual scenario')
    parser.add_argument('--root_dir', default="./Archives", type=str,
                        help='data directory name')
    parser.add_argument('--test_label', action='store_true', default=False,
                        help='define if we use task label at test')
    parser.add_argument('--dataset', default="MNIST", type=str, choices=["MNIST","mnist_fellowship"],
                        help='dataset name')

    args = parser.parse_args()
    args.root_dir = os.path.join(args.root_dir, args.dataset)
    if args.test_label:
        args.root_dir = os.path.join(args.root_dir, "MultiH")
    else:
        args.root_dir = os.path.join(args.root_dir, "SingleH")

    method_list = ["baseline", "ewc_diag", "rehearsal", "ewc_kfac", "ewc_diag_id","ogd"]
    method_list = ["baseline", "ewc_diag", "rehearsal", "ewc_kfac","ogd"]
    seed_list = [0]

    analysis = Continual_Analysis(args)
    analysis.save_latent()
    analysis.compute_Fishers()