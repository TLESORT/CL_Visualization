import pickle
import os
import torch
import numpy as np
from torch.utils import data
from nngeometry.layercollection import LayerCollection
from nngeometry.metrics import FIM
from nngeometry.object import PMatDiag
import abc

from copy import deepcopy


class Continual_Evaluation(abc.ABC):
    """ this class gives function to log for continual algorithms evaluation"""
    """ Log and Figure plotting should be clearly separate we can do on without the other """

    def __init__(self, args):
        self.vector_predictions_epoch_tr = np.zeros(0)
        self.vector_labels_epoch_tr = np.zeros(0)
        self.vector_task_labels_epoch_tr = np.zeros(0)

        self.vector_predictions_epoch_te = np.zeros(0)
        self.vector_labels_epoch_te = np.zeros(0)
        self.vector_task_labels_epoch_te = np.zeros(0)

        self.list_grad = {}
        self.list_loss = {}
        self.list_accuracies = {}
        self.list_accuracies_per_classes = {}
        self.list_latent = []  # we just need a simple list here
        self.list_weights = {}
        self.list_weights_dist = {}
        self.list_Fisher = []

    def init_log(self, ind_task_log):

        self.list_grad[ind_task_log] = []
        self.list_loss[ind_task_log] = []
        self.list_accuracies[ind_task_log] = []
        self.list_accuracies_per_classes[ind_task_log] = []
        self.list_weights[ind_task_log] = []
        self.list_weights_dist[ind_task_log] = []

        self.ref_model = deepcopy(self.model)


    def log_task(self, ind_task, model):
        model2save = deepcopy(model).cpu().state_dict()
        torch.save(model2save, os.path.join(self.log_dir, "Model_Task_{}.pth".format(ind_task)))

        self.log_latent(ind_task)

        F_diag, v0 = self.compute_last_layer_fisher(model, self.eval_tr_loader)
        self.list_Fisher.append(F_diag.get_diag().detach().cpu())

    def log_weights_dist(self, ind_task):

        # compute the l2 distance between current model and model at the beginning of the task
        dist_list = [torch.dist(p_current, p_ref) for p_current, p_ref in
                     zip(self.model.parameters(), self.ref_model.parameters())]
        dist = torch.tensor(dist_list).mean().detach().cpu().item()
        self.list_weights_dist[ind_task].append(dist)

    def _process_classes_logs(self, train):

        assert self.scenario_tr.nb_classes == self.scenario_te.nb_classes
        nb_classes = self.scenario_tr.nb_classes

        if train:
            vector_label = self.vector_labels_epoch_tr
            vector_preds = self.vector_predictions_epoch_tr
        else:
            vector_label = self.vector_labels_epoch_te
            vector_preds = self.vector_predictions_epoch_te

        classes_correctly_predicted = np.zeros(nb_classes)
        classes_wrongly_predicted = np.zeros(nb_classes)
        nb_instance_classes = np.zeros(nb_classes)

        for i in range(nb_classes):
            indexes_class = np.where(vector_label == i)[0]
            nb_instance_classes[i] = len(indexes_class)
            classes_correctly_predicted[i] = np.count_nonzero(vector_preds[indexes_class] == i)

            # number of predicted class i without the correctly predicted
            classes_wrongly_predicted[i] = len(np.where(vector_preds == i)[0]) - \
                                           classes_correctly_predicted[i]
            assert classes_wrongly_predicted[i] >= 0

        return np.array([classes_correctly_predicted, classes_wrongly_predicted, nb_instance_classes])

    def log_post_epoch_processing(self, ind_task, print_acc=False):

        # 1. processing for accuracy logs
        assert self.vector_predictions_epoch_tr.shape[0] == self.vector_labels_epoch_tr.shape[0]
        assert self.vector_predictions_epoch_te.shape[0] == self.vector_labels_epoch_te.shape[0]
        correct_tr = (self.vector_predictions_epoch_tr == self.vector_labels_epoch_tr).sum()
        correct_te = (self.vector_predictions_epoch_te == self.vector_labels_epoch_te).sum()
        nb_instances_tr = self.vector_labels_epoch_tr.shape[0]
        nb_instances_te = self.vector_labels_epoch_te.shape[0]

        # log correct prediction on nb instances for accuracy computation
        accuracy_infos = np.array([correct_tr, nb_instances_tr, correct_te, nb_instances_te])
        self.list_accuracies[ind_task].append(accuracy_infos)

        # 2. processing for accuracy per class logs
        class_infos_tr = self._process_classes_logs(train=True)
        class_infos_te = self._process_classes_logs(train=False)

        if print_acc:
            print("Train Accuracy: {} %".format(100.0 * correct_tr / nb_instances_tr))
            print("Test Accuracy: {} %".format(100.0 * correct_te / nb_instances_te))

        if self.verbose:
            classe_prediction, classe_wrong, classe_total=class_infos_te
            for i in range(self.scenario_tr.nb_classes):
                print("Task " + str(i) + "- Prediction :" + str(
                    classe_prediction[i] / classe_total[i]) + "% - Total :" + str(
                    classe_total[i]) + "- Wrong :" + str(classe_wrong[i]))

        self.list_accuracies_per_classes[ind_task].append(np.array([class_infos_tr, class_infos_te]))
        # Reinit log vector
        self.vector_predictions_epoch_tr = np.zeros(0)
        self.vector_labels_epoch_tr = np.zeros(0)
        self.vector_predictions_epoch_te = np.zeros(0)
        self.vector_labels_epoch_te = np.zeros(0)

    def log_iter(self, ind_task, model, loss, output, labels, task_labels, train=True):
        predictions = np.array(output.max(dim=1)[1].cpu())

        if train:
            self.vector_predictions_epoch_tr = np.concatenate([self.vector_predictions_epoch_tr, predictions])
            self.vector_labels_epoch_tr = np.concatenate([self.vector_labels_epoch_tr, labels.cpu().numpy()])
            self.vector_task_labels_epoch_tr  = np.concatenate([self.vector_task_labels_epoch_tr, task_labels])

            if model.fc2.weight.grad is not None:
                grad = model.fc2.weight.grad.clone().detach().cpu()
            else:
                # useful for first log before training
                grad = torch.zeros(model.fc2.weight.shape)

            self.list_grad[ind_task].append(grad)

            w = np.array(model.fc2.weight.data.detach().cpu().clone(), dtype=np.float16)
            b = np.array(model.fc2.bias.data.detach().cpu().clone(), dtype=np.float16)

            self.list_loss[ind_task].append(loss.data.clone().detach().cpu().item())
            self.list_weights[ind_task].append([w, b])

            self.log_weights_dist(ind_task)
        else:
            ## / ! \ we do not log loss for test, maybe one day....
            self.vector_predictions_epoch_te = np.concatenate([self.vector_predictions_epoch_te, predictions])
            self.vector_labels_epoch_te = np.concatenate([self.vector_labels_epoch_te, labels.cpu().numpy()])
            self.vector_task_labels_epoch_te  = np.concatenate([self.vector_task_labels_epoch_te, task_labels])

    def log_latent(self, ind_task):

        self.model.eval()

        latent_vectors = np.zeros([0, 50])
        y_vectors = np.zeros([0])

        for i_, (x_, y_, t_) in enumerate(self.eval_tr_loader):

            # data does not fit to the model if size<=1
            if x_.size(0) <= 1:
                continue
            x_ = x_.cuda()
            latent_vector = self.model(x_, latent_vector=True).detach().cpu()
            latent_vectors = np.concatenate([latent_vectors, latent_vector], axis=0)
            y_vectors = np.concatenate([y_vectors, np.array(y_)], axis=0)

            if len(y_vectors) >= 200:
                break
        latent_vectors = latent_vectors[:200]
        y_vectors = y_vectors[:200]
        self.list_latent.append([latent_vectors, y_vectors])


    def post_training_log(self):
        file_name = os.path.join(self.log_dir, "{}_loss.pkl".format(self.algo_name))
        with open(file_name, 'wb') as f:
            pickle.dump(self.list_loss, f, pickle.HIGHEST_PROTOCOL)

        file_name = os.path.join(self.log_dir, "{}_accuracies.pkl".format(self.algo_name))
        with open(file_name, 'wb') as f:
            pickle.dump(self.list_accuracies, f, pickle.HIGHEST_PROTOCOL)

        file_name = os.path.join(self.log_dir, "{}_accuracies_per_class.pkl".format(self.algo_name))
        with open(file_name, 'wb') as f:
            pickle.dump(self.list_accuracies_per_classes, f, pickle.HIGHEST_PROTOCOL)

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
                     representation=PMatDiag,
                     n_output=self.scenario_tr.nb_classes,
                     variant='classif_logits',
                     device='cuda')
        return F_diag, None
