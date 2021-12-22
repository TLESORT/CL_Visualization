import torch
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

from torch.utils.data import DataLoader

import torch.optim as optim
import torch.nn.functional as F
import os

from continuum.tasks import TaskSet, TaskType
from continuum.scenarios import create_subscenario

from utils import get_dataset, get_model, get_scenario, get_transform
from Encode.encode_utils import encode_scenario

import numpy as np

from eval import Continual_Evaluation


class Trainer(Continual_Evaluation):
    def __init__(self, config):

        super().__init__(config)

        self.lr = config.lr
        self.opt_name = config.opt_name
        self.weight_decay = config.weight_decay
        self.task_order = config.task_order
        self.momentum = config.momentum
        self.seed = config.seed
        self.root_dir = config.root_dir
        self.verbose = config.verbose
        self.dev = config.dev
        self.load_first_task = config.load_first_task
        self.batch_size = config.batch_size
        self.test_label = config.test_label
        self.masked_out = config.masked_out

        self.num_tasks = config.num_tasks
        self.increments = config.increments
        self.scenario_name = config.scenario_name
        self.subset = config.subset
        self.reset_opt = config.reset_opt

        self.algo_name = "baseline"
        self.fast = config.fast
        self.pretrained_on = config.pretrained_on
        self.OutLayer = config.OutLayer
        self.nb_epochs = config.nb_epochs
        self.non_differential_heads = ["SLDA", "MeanLayer", "MedianLayer", "KNN"]
        self.architecture = config.architecture

        dataset_train = get_dataset(self.data_dir, self.dataset, self.scenario_name, train=True)
        dataset_test = get_dataset(self.data_dir, self.dataset, self.scenario_name, train=False)

        self.transform_train = get_transform(self.dataset, architecture=self.architecture, train=True)
        self.transform_test = get_transform(self.dataset, architecture=self.architecture, train=False)

        self.scenario_tr = get_scenario(dataset_train, self.scenario_name, nb_tasks=self.num_tasks, increments=self.increments,
                                        transform=self.transform_train, config=config)

        self.scenario_te = get_scenario(dataset_test, self.scenario_name, nb_tasks=self.num_tasks, increments=self.increments,
                                        transform=self.transform_test, config=config)

        self.model = get_model(self.dataset,
                               self.scenario_tr,
                               self.pretrained_on,
                               self.test_label,
                               self.OutLayer,
                               self.name_algo,
                               model_dir=self.pmodel_dir,
                               architecture=self.architecture)
        self.model.cuda()

        self.finetuning = config.finetuning
        self.proj_drift_eval = config.proj_drift_eval
        self.data_encoded = False
        if (self.pretrained_on is not None) and not self.finetuning:
            # we replace the scenario data by feature vector from the pretrained model to save training time
            self.scenario_tr = encode_scenario(self.data_dir,
                                               self.scenario_tr,
                                               self.model,
                                               self.batch_size,
                                               name=f"encode_{config.dataset}_{config.architecture}"
                                                    f"_{config.pretrained_on}_{self.scenario_tr.nb_tasks}_train",
                                               train=True,
                                               dataset=self.dataset)
            self.scenario_te = encode_scenario(self.data_dir,
                                               self.scenario_te,
                                               self.model,
                                               self.batch_size,
                                               name=f"encode_{config.dataset}_{config.architecture}"
                                                    f"_{config.pretrained_on}_{self.scenario_te.nb_tasks}_test",
                                               train=False,
                                               dataset=self.dataset)
            self.data_encoded = True
            self.model.set_data_encoded(flag=True)
            self.transform_train = None
            self.transform_test = None

            assert self.scenario_tr.nb_tasks == self.num_tasks, \
                print(f"{self.scenario_tr.nb_tasks} vs {self.num_tasks}")

        if self.num_tasks > 1  or self.scenario_name=="SpuriousFeatures":
            # no need for mixing task in Spurious features and it create problems since there is not the same nb of task
            # in train and test
            # random permutation of task order
            if self.num_tasks > 1:
                self.scenario_tr = create_subscenario(self.scenario_tr, self.task_order)
            if self.scenario_te.nb_tasks > 1: # some test scenario have more task than train scenario
                test_task_order = self.task_order
                if self.scenario_name=="SpuriousFeatures":
                    test_task_order = np.concatenate([self.task_order,np.array([self.scenario_te.nb_tasks-1])])

                self.scenario_te = create_subscenario(self.scenario_te, test_task_order)

        if not self.data_encoded:
            for ind_task, task_set in enumerate(self.scenario_te):
                if task_set.data_type in [TaskType.IMAGE_ARRAY, TaskType.IMAGE_PATH]:
                    task_set.plot(self.sample_dir, f"samples_te_task_{ind_task}.png", 100,
                                  shape=self.model.data_shape)
            for ind_task, task_set in enumerate(self.scenario_tr):
                if task_set.data_type in [TaskType.IMAGE_ARRAY, TaskType.IMAGE_PATH]:
                    task_set.plot(self.sample_dir, f"samples_tr_task_{ind_task}.png", 100,
                                  shape=self.model.data_shape)

        self.num_classes = self.scenario_tr.nb_classes
        if not self.OutLayer in self.non_differential_heads:
            if self.opt_name=="SGD":
                self.opt = optim.SGD(params=self.model.parameters(), lr=self.lr, momentum=self.momentum)
            elif self.opt_name=="Adam":
                self.opt = optim.Adam(params=self.model.parameters(), lr=self.lr)
            else:
                raise NotImplementedError("this opt is not implemented here")
        else:
            self.opt = None

        self.eval_tr_loader = DataLoader(self.scenario_te[:], batch_size=self.batch_size, shuffle=True, num_workers=6)

        # shuffle should stay false for proj drift estimation
        self.eval_te_loader = DataLoader(self.scenario_te[:], batch_size=self.batch_size, shuffle=False, num_workers=6)

    def regularize_loss(self, model, loss):
        return loss

    def init_task(self, ind_task, task_set):

        # reset seed for consistency in results
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        if self.reset_opt and (not self.OutLayer in self.non_differential_heads):
            if self.opt_name=="SGD":
                self.opt = optim.SGD(params=self.model.parameters(), lr=self.lr, momentum=self.momentum)
            elif self.opt_name=="Adam":
                self.opt = optim.Adam(params=self.model.parameters(), lr=self.lr)
            else:
                raise NotImplementedError("this opt is not implemented here")

        if self.verbose: print("prepare subset")
        if self.subset is not None:
            # replace the full taskset by a subset of samples randomly selected
            nb_tot_samples = len(task_set)
            indexes = np.random.randint(0, nb_tot_samples, self.subset)
            x, y, t = task_set.get_raw_samples(indexes=indexes)
            task_set = TaskSet(x, y, t, trsf=task_set.trsf, data_type=task_set.data_type)

        print("Size Taskset")
        print(len(task_set))

        data_loader_tr = DataLoader(task_set,
                                    batch_size=self.batch_size,
                                    shuffle=True,
                                    num_workers=6)

        if not self.data_encoded: # if data is encoded we can not plot it
            task_set.plot(self.sample_dir, f"samples_task_{ind_task}.png",
                                  nb_samples=100,
                                  shape=[self.model.image_size, self.model.image_size, self.model.input_dim])

        if self.verbose: print("prepare log")
        if ind_task == 0:
            # log before training
            self.init_log(ind_task_log=ind_task)
            # if self.first_task_loaded -> we have already loaded test accuracy and train accuracy
            if not self.first_task_loaded:
                if self.verbose: print("test test")
                tuple_features_before_training = self.test(ind_task_log=ind_task, train=False)
                if self.verbose: print("test train")
                self.test(ind_task_log=ind_task, data_loader=data_loader_tr, train=True)
                self.log_post_epoch_processing(0, epoch=-1, tuple_features=tuple_features_before_training)
        return data_loader_tr

    def callback_task(self, ind_task, task_set):
        self.post_task_log(ind_task)

    def callback_epoch(self, ind_task, epoch):
        if self.OutLayer in self.non_differential_heads:
            self.model.update_head(epoch)

    def test(self, ind_task_log, data_loader=None, train=False, nb_embedding=200):


        if data_loader is None:
            data_loader = self.eval_te_loader

        np_embedding = np.zeros((0, self.model.features_size))
        np_classes = np.zeros(0)
        np_task_ids = np.zeros(0)

        for i_, (x_, y_, t_) in enumerate(data_loader):

            # data does not fit to the model if size<=1
            if x_.size(0) <= 1:
                continue
            y_ = y_.cuda()
            x_ = x_.cuda()

            self.model.eval()

            if not self.data_encoded:
                features = self.model.feature_extractor(x_)
            else:
                features = x_.view(x_.shape[0], -1)

            if self.proj_drift_eval and (not train):
                np_embedding = np.concatenate([np_embedding,features.detach().cpu()], axis=0)
                np_classes = np.concatenate([np_classes,np.array(y_.clone().cpu())], axis=0)
                np_task_ids = np.concatenate([np_task_ids,np.array(t_.clone().cpu())], axis=0)

            if self.test_label:
                output = self.model.head.forward_task(features, t_.long())
            else:
                output = self.model.get_last_layer()(features)

            loss = self.model.get_loss(output, y_, loss_func=F.cross_entropy)

            self.log_iter(ind_task_log, self.model, loss, output, y_, t_, train=train)

        if self.proj_drift_eval and (not train):
            np.random.seed(self.seed)
            selected_indexes = np.random.randint(np_embedding.shape[0], size=nb_embedding)
            np_embedding = np_embedding[selected_indexes]
            np_classes = np_classes[selected_indexes]
            np_task_ids = np_task_ids[selected_indexes]

        return (np_embedding, np_classes.astype(int), np_task_ids)

    def head_without_grad(self, x_, y_, t_, ind_task, epoch):

        if self.test_label:
            output = self.model.forward_task(x_, t_)
        else:
            output = self.model(x_)

        loss = self.model.get_loss(output,
                                   y_,
                                   loss_func=F.cross_entropy,
                                   masked=self.masked_out
                                   # we apply mask from task 1 because before there is no risk of forgetting
                                   )

        self.model.accumulate(x_, y_, epoch)
        return output, loss

    def head_with_grad(self, x_, y_, t_, ind_task, epoch):
        self.opt.zero_grad()
        if self.test_label:
            output = self.model.forward_task(x_, t_.long())
        else:
            output = self.model(x_)

        loss = self.model.get_loss(output,
                                   y_,
                                   loss_func=F.cross_entropy,
                                   masked=self.masked_out
                                   # we apply mask from task 1 because before there is no risk of forgetting
                                   )
        loss = self.regularize_loss(self.model, loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # clip gradient to avoid Nan
        self.optimizer_step(ind_task)

        return output, loss

    def one_task_training(self, ind_task, data_loader):

        for epoch in range(self.nb_epochs):
            if self.verbose: print(f"Epoch : {epoch}")
            self.model.train()
            if (self.subset is not None) and not (self.OutLayer in self.non_differential_heads):
                # we artificially augment the number of itereation for convergence purposes
                nb_run = int(50000 / self.subset)
            else:
                nb_run = 1
            for _ in range(nb_run):
                for i_, (x_, y_, t_) in enumerate(data_loader):
                    # data does not fit to the model if size<=1
                    if x_.size(0) <= 1:
                        continue

                    y_ = y_.cuda()
                    x_ = x_.cuda()

                    if self.OutLayer in self.non_differential_heads:
                        output, loss = self.head_without_grad(x_, y_, t_, ind_task, epoch)
                    else:
                        output, loss = self.head_with_grad(x_, y_, t_, ind_task, epoch)

                    self.log_iter(ind_task + 1, self.model, loss, output, y_, t_)
                    if self.dev: break

            self.callback_epoch(ind_task, epoch)
            tuple_post_epoch_features = self.test(ind_task_log=ind_task + 1)
            # we log and we print acc only for the last epoch
            self.log_post_epoch_processing(ind_task + 1,
                                           epoch=epoch,
                                           tuple_features=tuple_post_epoch_features,
                                           print_acc=(epoch == self.nb_epochs - 1))
            if self.dev: break

        return

    def optimizer_step(self, ind_task):
        self.opt.step()

    def continual_training(self):

        for task_id, task_set in enumerate(self.scenario_tr):
            print(f"Task {task_id}: Start")

            if self.verbose: print("init_task")
            data_loader = self.init_task(task_id, task_set)
            if task_id == 0 and self.first_task_loaded:
                # we loaded model and log from another experience for task 0
                continue
            if self.verbose: print("init_log")
            self.init_log(task_id + 1)  # after init task!
            if self.verbose: print("log_task")
            self.log_task(task_id, self.model)  # before training
            if self.verbose: print("one_task_training")
            self.one_task_training(task_id, data_loader)
            if self.verbose: print("callback_task")
            self.callback_task(task_id, task_set)

        # last log (we log  at the beginning of each task except for the last one)

        self.log_task(self.num_tasks, self.model)
        self.post_training_log()
