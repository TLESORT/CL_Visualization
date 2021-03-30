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

        self.dataset = args.dataset
        self.load_first_task=True
        self.first_task_loaded=False
        self.name_algo=args.name_algo

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

    def _can_load_first_task(self):
        model_weights_path = os.path.join(self.log_dir, f"Model_Task_0.pth")
        opt_weights_path = os.path.join(self.log_dir, f"Opt_Task_0.pth")
        return os.path.isfile(model_weights_path) and os.path.isfile(opt_weights_path) and self.name_algo != "ogd"

    def init_log(self, ind_task_log):

        self.list_grad[ind_task_log] = []
        self.list_loss[ind_task_log] = []
        self.list_accuracies[ind_task_log] = []
        self.list_accuracies_per_classes[ind_task_log] = []
        self.list_weights[ind_task_log] = []
        self.list_weights_dist[ind_task_log] = []

        self.ref_model = deepcopy(self.model)

        if ind_task_log==0 and self.load_first_task:
            if self._can_load_first_task():
                model_weights_path = os.path.join(self.log_dir, f"Model_Task_{ind_task_log}.pth")
                opt_weights_path = os.path.join(self.log_dir, f"Opt_Task_{ind_task_log}.pth")

                pretrained_weights = torch.load(model_weights_path)
                self.model.load_state_dict(pretrained_weights)

                opt_dict = torch.load(opt_weights_path)
                self.opt.load_state_dict(opt_dict)
                self.load_log(ind_task_log)
                self.first_task_loaded=True
                print(" EVERYTHING HAVE BEEN LOADED SUCCESSFULLY")
            else:
                print("No file to load continue training normally")


    def log_task(self, ind_task, model):
        torch.save(model.state_dict(), os.path.join(self.log_dir, f"Model_{self.name_algo}_Task_{ind_task}.pth"))
        if not self.fast:
            self.log_latent(ind_task)

            F_diag, v0 = self.compute_last_layer_fisher(model, self.eval_tr_loader)
            self.list_Fisher.append(F_diag.get_diag().detach().cpu())


    def post_task_log(self, ind_task):
        if ind_task==0 and self.name_algo != "ogd":
            # we will save the state of the training to save time for other experiments

            torch.save(self.model.state_dict(), os.path.join(self.log_dir, f"Model_Task_{ind_task}.pth"))
            # log optimizer
            torch.save(self.opt.state_dict(), os.path.join(self.log_dir, f"Opt_Task_{ind_task}.pth"))

            # save log from first tasks
            self.post_training_log(ind_task)



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
            acc_tr = 100.0 * correct_tr / nb_instances_tr
            acc_te = 100.0 * correct_te / nb_instances_te
            print(f"Train Accuracy: {acc_tr} %")
            print(f"Test Accuracy: {acc_te} %")

        if self.verbose:
            classe_prediction, classe_wrong, classe_total = class_infos_te
            for i in range(self.scenario_tr.nb_classes):
                print(f"Task {i} - Prediction : {classe_prediction[i] / classe_total[i]} % "
                      f"- Total :{classe_total[i]}- Wrong :{classe_wrong[i]}")

        self.list_accuracies_per_classes[ind_task].append(np.array([class_infos_tr, class_infos_te]))
        # Reinit log vector
        self.vector_predictions_epoch_tr = np.zeros(0)
        self.vector_labels_epoch_tr = np.zeros(0)
        self.vector_predictions_epoch_te = np.zeros(0)
        self.vector_labels_epoch_te = np.zeros(0)

    def _multihead_predictions(self, output, labels, task_labels):

        assert self.scenario_name == "Disjoint"

        list_preds = []
        for x, y, t in zip(output, labels, task_labels):
            # / ! \ data are processed one by one here

            label = y.cpu().item()

            # the output is a vector of size equal to the total number of classes
            # all value are zeros exept those corresponding to the correct head
            ind_mask = self.model.classes_heads[label]
            head_mask = self.model.heads_mask[ind_mask.long()]
            inds_mask = torch.nonzero(head_mask)
            local_pred = x[inds_mask].argmax().item()
            pred = inds_mask[local_pred].item()
            list_preds.append(pred)

        assert len(list_preds) == len(labels)

        return np.array(list_preds)



    def log_iter(self, ind_task, model, loss, output, labels, task_labels, train=True):

        if not self.test_label:
            predictions = np.array(output.max(dim=1)[1].cpu())
        else:
            predictions = self._multihead_predictions(output, labels, task_labels)

        if train:
            self.vector_predictions_epoch_tr = np.concatenate([self.vector_predictions_epoch_tr, predictions])
            self.vector_labels_epoch_tr = np.concatenate([self.vector_labels_epoch_tr, labels.cpu().numpy()])
            self.vector_task_labels_epoch_tr = np.concatenate([self.vector_task_labels_epoch_tr, task_labels])

            if not self.fast:
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
            self.vector_task_labels_epoch_te = np.concatenate([self.vector_task_labels_epoch_te, task_labels])

    def log_latent(self, ind_task):
        self.model.eval()

        latent_vectors = np.zeros([0, 50])
        y_vectors = np.zeros([0])
        t_vectors = np.zeros([0])

        for i_, (x_, y_, t_) in enumerate(self.eval_tr_loader):

            # data does not fit to the model if size<=1
            if x_.size(0) <= 1:
                continue
            x_ = x_.cuda()
            latent_vector = self.model(x_, latent_vector=True).detach().cpu()
            latent_vectors = np.concatenate([latent_vectors, latent_vector], axis=0)
            y_vectors = np.concatenate([y_vectors, np.array(y_)], axis=0)
            t_vectors = np.concatenate([t_vectors, np.array(t_)], axis=0)

            if len(y_vectors) >= 200:
                break
        latent_vectors = latent_vectors[:200]
        y_vectors = y_vectors[:200]
        t_vectors = t_vectors[:200]
        self.list_latent.append([latent_vectors, y_vectors, t_vectors])

    def load_log(self, ind_task=None):
        assert ind_task==0, print("The code is not made yet for ind task <> 0")
        name = f"checkpoint_{ind_task}"
        file_name = os.path.join(self.log_dir, f"{name}_loss.pkl")
        with open(file_name, 'rb') as fp:
            self.list_los = pickle.load(fp)

        file_name = os.path.join(self.log_dir, f"{name}_accuracies.pkl")
        with open(file_name, 'rb') as fp:
            self.list_accuracies = pickle.load(fp)

        file_name = os.path.join(self.log_dir, f"{name}_accuracies_per_class.pkl")
        with open(file_name, 'rb') as fp:
            self.list_accuracies_per_classes = pickle.load(fp)

        if not self.fast:
            file_name = os.path.join(self.log_dir, f"{name}_grad.pkl")
            with open(file_name, 'rb') as fp:
                self.list_grad = pickle.load(fp)

            file_name = os.path.join(self.log_dir, f"{name}_weights.pkl")
            with open(file_name, 'rb') as fp:
                self.list_weights = pickle.load(fp)

            file_name = os.path.join(self.log_dir, f"{name}_dist.pkl")
            with open(file_name, 'rb') as fp:
                self.list_weights_dist = pickle.load(fp)


            file_name = os.path.join(self.log_dir, f"{name}_Latent.pkl")
            with open(file_name, 'rb') as fp:
                self.list_latent = pickle.load(fp)

    def post_training_log(self, ind_task=None):

        if ind_task is None:
            name = self.name_algo
        else:
            assert ind_task==0, print("The code is not made yet for ind task <> 0")
            name = f"checkpoint_{ind_task}"

        file_name = os.path.join(self.log_dir, f"{name}_loss.pkl")
        with open(file_name, 'wb') as f:
            pickle.dump(self.list_loss, f, pickle.HIGHEST_PROTOCOL)

        file_name = os.path.join(self.log_dir, f"{name}_accuracies.pkl")
        with open(file_name, 'wb') as f:
            pickle.dump(self.list_accuracies, f, pickle.HIGHEST_PROTOCOL)

        file_name = os.path.join(self.log_dir, f"{name}_accuracies_per_class.pkl")
        with open(file_name, 'wb') as f:
            pickle.dump(self.list_accuracies_per_classes, f, pickle.HIGHEST_PROTOCOL)

        if not self.fast:
            file_name = os.path.join(self.log_dir, f"{name}_grad.pkl")
            with open(file_name, 'wb') as f:
                pickle.dump(self.list_grad, f, pickle.HIGHEST_PROTOCOL)

            file_name = os.path.join(self.log_dir, f"{name}_weights.pkl")
            with open(file_name, 'wb') as f:
                pickle.dump(self.list_weights, f, pickle.HIGHEST_PROTOCOL)

            file_name = os.path.join(self.log_dir, f"{name}_dist.pkl")
            with open(file_name, 'wb') as f:
                pickle.dump(self.list_weights_dist, f, pickle.HIGHEST_PROTOCOL)


            file_name = os.path.join(self.log_dir, f"{name}_Latent.pkl")
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
