import pickle
import os
import torch
import numpy as np
from numpy import linalg as LA
import abc
from multiprocessing import Pool, cpu_count

from Plot.plot_utils import angle_between
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
        self.offline = config.offline

        self.data_dir = config.data_dir
        self.pmodel_dir = config.pmodel_dir
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir

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

        if not self.fast and not self.OutLayer == "OriginalWeightNorm":
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
            if self.OutLayer == "KNN":
                knnPickle = open(os.path.join(self.log_dir, f"Head_{self.name_algo}_Task_{ind_task}.pkl"), 'wb')
                pickle.dump(model.get_last_layer().neigh, knnPickle)
            else:
                torch.save(model.get_last_layer().state_dict(),
                           os.path.join(self.log_dir, f"Head_{self.name_algo}_Task_{ind_task}.pth"))
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

        if not self.dev and not self.fast and (not self.OutLayer in self.non_differential_heads):
            layer = self.model.get_last_layer()
            w = np.array(layer.weight.data.detach().cpu().clone(), dtype=np.float16)
            wandb.log({"weights output": wandb.Histogram(w)})
            classes_ids = np.arange(self.num_classes)
            if hasattr(layer, 'bias') and layer.bias is not None:
                b = np.array(layer.bias.data.detach().cpu().clone(), dtype=np.float16)
                data_b = [[s, class_id] for s, class_id in zip(b, classes_ids)]
                table_b = wandb.Table(data=data_b, columns=["bias", "Class"])
                wandb.log({f"Bias Task {ind_task}": wandb.plot.bar(table_b, "Class", "bias",
                                                                   title=f"Bias Bars Task {ind_task}")})

            norm_mat = np.zeros(self.num_classes)

            for j in range(self.num_classes):
                norm_mat[j] = LA.norm(w[j, :])

            data_norm = [[s, class_id] for s, class_id in zip(norm_mat, classes_ids)]
            table_norm = wandb.Table(data=data_norm, columns=["Norm", "Class"])
            wandb.log({f"Norm Task {ind_task}": wandb.plot.bar(table_norm, "Class", "Norm",
                                                               title=f"Norm  Bars Task {ind_task}")})

    def log_weights_dist(self, ind_task):

        # compute the l2 distance between current model and model at the beginning of the task
        if not self.OutLayer == "OriginalWeightNorm":
            dist_list = [torch.dist(p_current, p_ref) for p_current, p_ref in
                         zip(self.model.parameters(), self.ref_model.parameters())]
            dist = torch.tensor(dist_list).mean().detach().cpu().item()
            self.list_weights_dist[ind_task].append(dist)
        else:
            print("Log of weight dist is not implemented for OriginalWeightNorm")

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

    def log_post_epoch_processing(self, ind_task, epoch, tuple_features, print_acc=False):

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

        accuracy_tr = (1.0 * correct_tr) / nb_instances_tr
        accuracy_te = (1.0 * correct_te) / nb_instances_te

        if not (self.dev or self.offline):

            if self.dataset not in ["Core50", "Core10Lifelong", "Core10Mix"]:
                # the test set of core50 is not compatible with an evaluation by task especially in Lifelong setting

                assert self.vector_task_labels_epoch_te.shape[0] == self.vector_labels_epoch_te.shape[0]

                list_tasks = np.unique(self.vector_task_labels_epoch_te).astype(int)

                assert len(list_tasks) == self.num_tasks, print(f"{len(list_tasks)} vs {self.num_tasks}")

                for i in list_tasks:
                    indexes = np.where(self.vector_task_labels_epoch_te == list_tasks[i])[0]
                    vector_labels_epoch_te_task = self.vector_labels_epoch_te[indexes]
                    vector_predictions_epoch_te_task = self.vector_predictions_epoch_te[indexes]
                    correct_te_task = (vector_predictions_epoch_te_task == vector_labels_epoch_te_task).sum()
                    assert len(indexes) == vector_labels_epoch_te_task.shape[0], \
                        print(f'{len(indexes)} vs {vector_labels_epoch_te_task.shape[0]}')
                    accuracy_te_task = (1.0 * correct_te_task) / len(indexes)

                    print(f'test accuracy task {list_tasks[i]}: {accuracy_te_task}, epoch - {self.nb_tot_epoch},task {ind_task}')

                    wandb.log({f'test accuracy task {list_tasks[i]}': accuracy_te_task, 'epoch': self.nb_tot_epoch,
                               'task': ind_task})

            wandb.log({'train accuracy': accuracy_tr,
                       'test accuracy': accuracy_te,
                       'epoch': self.nb_tot_epoch,
                       'task': ind_task})

        if self.proj_drift_eval and (self.pretrained_on is not None) and self.finetuning:

            np_features, np_classes, np_task_ids = tuple_features[0], tuple_features[1], tuple_features[2]
            assert len(np_features) > 0
            assert len(np_features) == len(np_classes)
            if not (ind_task == 0 and epoch == -1):
                assert np.all(self.ref_classes == np_classes)

            weight_matrix = np.array(self.model.get_last_layer().weight.data.detach().cpu().clone())

            if ind_task == 0 and epoch == -1:  # epoch == -1 means before starting continual training
                # estimate initial for tot_proj_drift

                self.ref_features = {}
                self.norm_drifts = {}
                self.angle_drift = {}
                self.ref_classes = np_classes
                self.init_features = np_features
                self.ref_features = np.zeros(0, np_features.shape[0], np_features.shape[1])
            else:
                if epoch == self.nb_epochs - 1:  # just before next task we prepare ref features
                    self.ref_features = np.concatenate([self.ref_features, np_features], axis=0)

                if epoch == 0:
                    self.norm_drifts[ind_task] = []
                    self.angle_drift[ind_task] = []

                # mesure task_proj_drift and tot_proj_drift

                # ref
                diff_representation = np_features - self.init_features

                # angles
                with Pool(min(8, cpu_count())) as p:
                    list_repr_angles = list(
                        p.starmap(angle_between, zip(list(self.init_features), list(np_features))))

                np_repr_angles = np.array(list_repr_angles)

                with Pool(min(8, cpu_count())) as p:
                    list_angles = list(
                        p.starmap(angle_between, zip(list(diff_representation), list(weight_matrix[np_classes]))))

                norm_drift = LA.norm(diff_representation, axis=1)

                np_angles = np.array(list_angles)

                self.norm_drifts[ind_task].append(norm_drift)
                self.angle_drift[ind_task].append(np_angles)

                for i in range(self.num_classes):
                    indexes = np.where(np_classes == i)[0]
                    norm_class = norm_drift[indexes]
                    np_angles_class = np_angles[indexes]
                    np_repr_angles_class = np_repr_angles[indexes]
                    wandb.log({f'delta norm class {i}': norm_class.mean(),
                               f'delta angle class {i}': np_angles_class.mean(),
                               f'delta repr angle class {i}': np_repr_angles_class.mean(),
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
        self.vector_task_labels_epoch_tr = np.zeros(0)
        self.vector_predictions_epoch_te = np.zeros(0)
        self.vector_labels_epoch_te = np.zeros(0)
        self.vector_task_labels_epoch_te = np.zeros(0)

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
            output = output.mean(1)

        if not self.test_label:
            predictions = np.array(output.max(dim=1)[1].cpu())
        else:
            predictions = self._multihead_predictions(output, labels, task_labels)

        if not (self.dev or self.offline):
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

                if not (
                        self.name_algo == "ogd" or self.OutLayer == "SLDA" or self.OutLayer == "KNN" or "MIMO" in self.OutLayer):
                    if model.get_last_layer().weight.grad is not None:
                        grad = model.get_last_layer().weight.grad.clone().detach().cpu()
                    else:
                        # useful for first log before training
                        grad = torch.zeros(model.get_last_layer().weight.shape)

                    self.list_grad[ind_task].append(grad)

                    w = np.array(model.get_last_layer().weight.data.detach().cpu().clone(), dtype=np.float16)
                    if self.OutLayer == "Linear":
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
