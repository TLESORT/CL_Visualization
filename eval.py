import pickle
import os
import torch
import numpy as np
import abc

from copy import deepcopy
import wandb


class Continual_Evaluation(abc.ABC):
    """ this class gives function to log for continual algorithms evaluation"""
    """ Log and Figure plotting should be clearly separate we can do on without the other """

    def __init__(self, config):

        self.dataset = config.dataset
        self.load_first_task = config.load_first_task
        self.first_task_loaded = False  # flag
        self.name_algo = config.name_algo
        self.nb_tot_epoch = None

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

        if ind_task_log == 0 and self.load_first_task:
            if self._can_load_first_task():
                model_weights_path = os.path.join(self.log_dir, f"Model_Task_{ind_task_log}.pth")
                opt_weights_path = os.path.join(self.log_dir, f"Opt_Task_{ind_task_log}.pth")

                pretrained_weights = torch.load(model_weights_path)
                self.model.load_state_dict(pretrained_weights)

                opt_dict = torch.load(opt_weights_path)
                self.opt.load_state_dict(opt_dict)
                self.load_log(ind_task_log)
                self.first_task_loaded = True
                print(" EVERYTHING HAVE BEEN LOADED SUCCESSFULLY")
            else:
                print("No file to load continue training normally")

    def log_task(self, ind_task, model):
        if self.pretrained_on:
            torch.save(model.get_last_layer().state_dict(), os.path.join(self.log_dir, f"Head_{self.name_algo}_Task_{ind_task}.pth"))
        else:
            torch.save(model.state_dict(), os.path.join(self.log_dir, f"Model_{self.name_algo}_Task_{ind_task}.pth"))

    def post_task_log(self, ind_task):
        if ind_task == 0 and self.name_algo != "ogd":
            # we will save the state of the training to save time for other experiments

            torch.save(self.model.state_dict(), os.path.join(self.log_dir, f"Model_Task_{ind_task}.pth"))
            # log optimizer if one
            if self.opt is not None:
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

        assert self.scenario_tr.nb_classes == self.scenario_te.nb_classes, \
            print(f"Train {self.scenario_tr.nb_classes} - Test {self.scenario_te.nb_classes}")
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

    def log_post_epoch_processing(self, ind_task, epoch, print_acc=False):

        if self.nb_tot_epoch is None:
            self.nb_tot_epoch = epoch
        else:
            self.nb_tot_epoch += 1

        # 1. processing for accuracy logs
        assert self.vector_predictions_epoch_tr.shape[0] == self.vector_labels_epoch_tr.shape[0]
        assert self.vector_predictions_epoch_te.shape[0] == self.vector_labels_epoch_te.shape[0]
        correct_tr = (self.vector_predictions_epoch_tr == self.vector_labels_epoch_tr).sum()
        correct_te = (self.vector_predictions_epoch_te == self.vector_labels_epoch_te).sum()
        nb_instances_tr = self.vector_labels_epoch_tr.shape[0]
        nb_instances_te = self.vector_labels_epoch_te.shape[0]

        accuracy_tr = correct_tr / (1.0 * nb_instances_tr)
        accuracy_te = correct_te / (1.0 * nb_instances_te)

        if not self.dev:
            wandb.log({'train accuracy': accuracy_tr,
                       'test accuracy': accuracy_te,
                       'epoch': self.nb_tot_epoch,
                       'task': ind_task})

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
            ind_mask = self.model.head.classes_heads[label]
            head_mask = self.model.head.heads_mask[ind_mask.long()]
            inds_mask = torch.nonzero(head_mask)
            local_pred = x[inds_mask].argmax().item()
            pred = inds_mask[local_pred].item()
            list_preds.append(pred)

        assert len(list_preds) == len(labels)

        return np.array(list_preds)

    def log_iter(self, ind_task, model, loss, output, labels, task_labels, train=True):

        if "MIMO_" in self.OutLayer:
            # the prediction average output
            output=output.mean(1)

        if not self.test_label:
            predictions = np.array(output.max(dim=1)[1].cpu())
        else:
            predictions = self._multihead_predictions(output, labels, task_labels)

        if not self.dev:
            if train:
                wandb.log({'train loss': loss})
            else:
                wandb.log({'test loss': loss})

        if train:
            self.vector_predictions_epoch_tr = np.concatenate([self.vector_predictions_epoch_tr, predictions])
            self.vector_labels_epoch_tr = np.concatenate([self.vector_labels_epoch_tr, labels.cpu().numpy()])
            self.vector_task_labels_epoch_tr = np.concatenate([self.vector_task_labels_epoch_tr, task_labels])

            if not self.fast:


                self.list_loss[ind_task].append(loss.data.clone().detach().cpu().item())

                if not (self.name_algo=="ogd" or self.OutLayer=="SLDA" or  self.OutLayer=="KNN" or "MIMO" in self.OutLayer):
                    if model.get_last_layer().weight.grad is not None:
                        grad = model.get_last_layer().weight.grad.clone().detach().cpu()
                    else:
                        # useful for first log before training
                        grad = torch.zeros(model.get_last_layer().weight.shape)

                    self.list_grad[ind_task].append(grad)

                    w = np.array(model.get_last_layer().weight.data.detach().cpu().clone(), dtype=np.float16)
                    if self.OutLayer=="Linear":
                        b = np.array(model.get_last_layer().bias.data.detach().cpu().clone(), dtype=np.float16)
                        self.list_weights[ind_task].append([w, b])
                    else:
                        self.list_weights[ind_task].append(w)
                else:
                    # todo
                    # the weights of the output layer are split, it's complicated
                    pass
                self.log_weights_dist(ind_task)
        else:
            ## / ! \ we do not log loss for test, maybe one day....
            self.vector_predictions_epoch_te = np.concatenate([self.vector_predictions_epoch_te, predictions])
            self.vector_labels_epoch_te = np.concatenate([self.vector_labels_epoch_te, labels.cpu().numpy()])
            self.vector_task_labels_epoch_te = np.concatenate([self.vector_task_labels_epoch_te, task_labels])

    def load_log(self, ind_task=None):
        assert ind_task == 0, print("The code is not made yet for ind task <> 0")
        name = f"checkpoint_{ind_task}"
        file_name = os.path.join(self.log_dir, f"{name}_loss.pkl")
        with open(file_name, 'rb') as fp:
            self.list_loss = pickle.load(fp)

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

    def post_training_log(self, ind_task=None):

        if ind_task is None:
            name = self.name_algo
        else:
            assert ind_task == 0, print("The code is not made yet for ind task <> 0")
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
