import torch
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

from torch.utils.data import DataLoader

import torch.optim as optim
import torch.nn.functional as F
import os

from utils import get_dataset, get_model, get_scenario, get_transform
from Encode.encode_utils import encode_scenario

import numpy as np

from eval import Continual_Evaluation


class Trainer(Continual_Evaluation):
    def __init__(self, args):

        super().__init__(args)

        self.lr = args.lr
        self.seed = args.seed
        self.root_dir = args.root_dir
        self.verbose = args.verbose
        self.dev = args.dev
        self.load_first_task = args.load_first_task
        self.batch_size = args.batch_size
        self.test_label = args.test_label
        self.masked_out = args.masked_out
        self.num_tasks = args.num_tasks
        self.scenario_name = args.scenario_name

        self.data_dir = args.data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        self.log_dir = os.path.join(self.root_dir, "Logs", self.scenario_name)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.sample_dir = os.path.join(self.root_dir, "Samples")
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)

        self.algo_name = "baseline"
        self.fast = args.fast
        self.pretrained_on = args.pretrained_on
        self.OutLayer = args.OutLayer
        self.nb_epochs = args.nb_epochs


        dataset_train = get_dataset(self.data_dir, args.dataset, self.scenario_name, train=True)
        dataset_test = get_dataset(self.data_dir, args.dataset, self.scenario_name, train=False)

        self.transform_train = get_transform(self.dataset, train=True)
        self.transform_test = get_transform(self.dataset, train=True)


        print()
        self.scenario_tr = get_scenario(dataset_train, self.scenario_name, nb_tasks=self.num_tasks, transform=self.transform_train)
        self.scenario_te = get_scenario(dataset_test, self.scenario_name, nb_tasks=self.num_tasks, transform=self.transform_test)

        self.model = get_model(self.dataset, self.scenario_tr, self.pretrained_on, self.test_label, self.OutLayer, self.name_algo)
        self.model.cuda()

        self.finetuning = False
        if (self.pretrained_on is not None) and self.finetuning==False:
            # we replace the scenario data by feature vector from the pretrained model to save training time
            self.scenario_tr = encode_scenario(self.data_dir,
                                               self.scenario_tr,
                                               self.model,
                                               name=f"encode_{args.dataset}_{self.scenario_tr.nb_tasks}_train")
            self.scenario_te = encode_scenario(self.data_dir,
                                               self.scenario_te,
                                               self.model,
                                               name=f"encode_{args.dataset}_{self.scenario_te.nb_tasks}_test")


        self.num_classes = self.scenario_tr.nb_classes
        self.classes_mask = torch.eye(self.num_classes).cuda()

        self.opt = optim.SGD(params=self.model.parameters(), lr=self.lr, momentum=args.momentum)

        self.eval_tr_loader = DataLoader(self.scenario_te[:], batch_size=self.batch_size, shuffle=True, num_workers=6)

        self.eval_te_loader = DataLoader(self.scenario_te[:], batch_size=self.batch_size, shuffle=True, num_workers=6)

    def regularize_loss(self, model, loss):
        return loss

    def init_task(self, ind_task, task_set):

        # reset seed for consistency in results
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        data_loader_tr = DataLoader(task_set,
                                    batch_size=self.batch_size,
                                    shuffle=True,
                                    num_workers=6)
        if ind_task == 0:
            # log before training
            self.init_log(ind_task_log=ind_task)
            # if self.first_task_loaded -> we have already loaded test accuracy and train accuracy
            if not self.first_task_loaded:
                self.test(ind_task_log=ind_task, train=False)
                self.test(ind_task_log=ind_task, data_loader=data_loader_tr, train=True)
                self.log_post_epoch_processing(0)
        return data_loader_tr

    def callback_task(self, ind_task, task_set):
        self.post_task_log(ind_task)

    def test(self, ind_task_log, data_loader=None, train=False):

        if data_loader is None:
            data_loader = self.eval_te_loader
        for i_, (x_, y_, t_) in enumerate(data_loader):

            # data does not fit to the model if size<=1
            if x_.size(0) <= 1:
                continue
            y_ = y_.cuda()
            x_ = x_.cuda()

            self.model.eval()
            if self.test_label:
                output = self.model.forward_task(x_, t_)
            else:
                output = self.model(x_)
            loss = F.cross_entropy(output, y_)
            self.log_iter(ind_task_log, self.model, loss, output, y_, t_, train=train)

    def one_task_training(self, ind_task, data_loader):

        for epoch in range(self.nb_epochs):
            for i_, (x_, y_, t_) in enumerate(data_loader):

                # data does not fit to the model if size<=1
                if x_.size(0) <= 1:
                    continue

                y_ = y_.cuda()
                x_ = x_.cuda()

                self.model.train()
                self.opt.zero_grad()
                if self.test_label:
                    output = self.model.forward_task(x_, t_)
                else:
                    output = self.model(x_)

                if self.masked_out and ind_task > 0:
                    masked_output = torch.mul(output, self.classes_mask[y_])
                    loss = F.cross_entropy(masked_output, y_)
                else:
                    loss = F.cross_entropy(output, y_)

                assert output.shape[0] == y_.shape[0]

                loss = self.regularize_loss(self.model, loss)

                if self.OutLayer != "SLDA":
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # clip gradient to avoid Nan
                    self.optimizer_step(ind_task)
                else:
                    self.model.update_head(x_, y_)
                self.log_iter(ind_task + 1, self.model, loss, output, y_, t_)

                if self.dev: break

            self.test(ind_task_log=ind_task + 1)
            # we log and we print acc only for the last epoch
            self.log_post_epoch_processing(ind_task + 1, print_acc=(epoch == self.nb_epochs - 1))
            if self.dev: break

        return

    def optimizer_step(self, ind_task):
        self.opt.step()

    def continual_training(self):

        for task_id, task_set in enumerate(self.scenario_tr):
            print("Task {}: Start".format(task_id))

            data_loader = self.init_task(task_id, task_set)
            if task_id==0 and self.first_task_loaded:
                # we loaded model and log from another experience for task 0
                continue
            self.init_log(task_id + 1)  # after init task!
            self.log_task(task_id, self.model)  # before training
            self.one_task_training(task_id, data_loader)
            self.callback_task(task_id, task_set)

        # last log (we log  at the beginning of each task except for the last one)

        self.log_task(self.num_tasks, self.model)
        self.post_training_log()
