
import pickle
import os
import torch
from torch.utils import data

from copy import deepcopy

class Continual_Evaluation(object):
    """ this class gives function to log for continual algorithms evaluation"""
    """ Log and Figure plotting should be clearly separate we can do on without the other """
    def __init__(self, args):
        print("Abstract class")

    def init_log(self, ind_task):
        if ind_task == 0:
            self.list_loss = {}
            self.list_grad = {}
            self.list_latent = {}
            self.list_weights = {}
            self.list_weights_dist = {}
            self.list_Fisher = []

        if ind_task == self.num_tasks:
            # we are only interted to latent vector here the training is over
            self.list_latent[ind_task] = []
        else:
            self.list_grad[ind_task] = []
            self.list_loss[ind_task] = []
            self.list_latent[ind_task] = []
            self.list_weights[ind_task] = []
            self.list_weights_dist[ind_task] = []

        self.ref_model = deepcopy(self.model)

    def log_weights_dist(self, ind_task):

        # compute the l2 distance between current model and model at the beginning of the task
        dist_list = [torch.dist(p_current, p_ref) for p_current, p_ref in zip(self.model.parameters(), self.ref_model.parameters())]
        dist = torch.tensor(dist_list).mean().item()
        self.list_weights_dist[ind_task].append(dist)


    def log_iter(self, ind_task, model, loss):

        grad = model.fc2.weight.grad.clone().cpu()
        self.list_grad[ind_task].append(grad)
        w = model.fc2.weight.data.cpu().clone()
        b = model.fc2.bias.data.cpu().clone()

        self.list_loss[ind_task].append(loss.data.clone().cpu().item())
        self.list_weights[ind_task].append([w, b])

        self.log_weights_dist(ind_task)

    def log_latent(self, ind_task):

        self.model.eval()

        for i_, (x_, t_) in enumerate(self.eval_tr_loader):

            # data does not fit to the model if size<=1
            if x_.size(0) <= 1:
                continue
            x_ = x_.cuda()
            self.list_latent[ind_task].append([self.model(x_, latent_vector=True).cpu(), t_])


    def log_task(self, ind_task, model):

        torch.save(self.model.state_dict(), os.path.join(self.log_dir, "Model_Task_{}.pth".format(ind_task)))

        self.log_latent(ind_task)

        F_diag, v0 = self.compute_last_layer_fisher(self.model, self.eval_tr_loader)
        self.list_Fisher.append(F_diag.get_diag())

    def post_training_log(self):
        file_name = os.path.join(self.log_dir, "{}_loss.pkl".format(self.algo_name))
        with open(file_name, 'wb') as f:
            pickle.dump(self.list_loss, f, pickle.HIGHEST_PROTOCOL)

        file_name = os.path.join(self.log_dir, "{}_grad.pkl".format(self.algo_name))
        with open(file_name, 'wb') as f:
            pickle.dump(self.list_grad, f, pickle.HIGHEST_PROTOCOL)

        file_name = os.path.join(self.log_dir, "{}_weights.pkl".format(self.algo_name))
        with open(file_name, 'wb') as f:
            pickle.dump(self.list_weights, f, pickle.HIGHEST_PROTOCOL)

        file_name = os.path.join(self.log_dir, "{}_dist.pkl".format(self.algo_name))
        with open(file_name, 'wb') as f:
            pickle.dump(self.list_weights_dist, f, pickle.HIGHEST_PROTOCOL)

        file_name = os.path.join(self.log_dir, "{}_Fishers.pkl".format(self.algo_name))
        with open(file_name, 'wb') as f:
            pickle.dump(self.list_Fisher, f, pickle.HIGHEST_PROTOCOL)

        file_name = os.path.join(self.log_dir, "{}_Latent.pkl".format(self.algo_name))
        with open(file_name, 'wb') as f:
            pickle.dump(self.list_latent, f, pickle.HIGHEST_PROTOCOL)

    def compute_last_layer_fisher(self, model, fisher_loader):

        layer_collection_all_layers = LayerCollection.from_model(model)
        layer_collection_last_layer = LayerCollection()
        layer_collection_last_layer.add_layer(*list(layer_collection_all_layers.layers.items())[-1])

        F_diag = FIM(layer_collection=layer_collection_last_layer,
                     model=model,
                     loader=fisher_loader,
                     representation=PSpaceDiag,
                     n_output=10,
                     variant='classif_logits',
                     device='cuda')
        return F_diag, None