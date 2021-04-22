import torch.nn as nn
import torch
import numpy as np

from Models.model_utils import get_Output_layer

class NNHead(nn.Module):
    def __init__(self, input_size, num_classes=10, classes_per_tasks=None, LayerType="Linear", method='baseline'):
        super(NNHead, self).__init__()
        self.input_size = input_size
        self.method=method
        self.num_classes = num_classes
        self.LayerType=LayerType

        if not (classes_per_tasks is None):
            assert len(np.unique(classes_per_tasks))==self.num_classes

            self.num_head=len(classes_per_tasks)

            # vector that for each class gives the correct head index
            self.classes_heads=torch.zeros(self.num_classes)
            self.heads_mask=torch.zeros(self.num_head, self.num_classes).cuda()


            self.classes_per_tasks = classes_per_tasks
            if self.classes_per_tasks is not None:
                self.multi_heads = True
                self.list_heads = []
                # we suppose no overlaps in heads
                for i, classes in enumerate(self.classes_per_tasks):
                    for _class in classes:
                        # we register the head associated with a given class
                        self.classes_heads[_class]=i
                        self.heads_mask[i, _class]=1
                    dim = len(classes)
                    if self.method=="ogd":
                        self.list_heads.append(get_Output_layer(self.LayerType, self.input_size, dim))
                    else:
                        # Normal Output Layer
                        self.layer = get_Output_layer(self.LayerType, self.input_size, dim)

        else:
            # Normal Output Layer
            self.layer = get_Output_layer(self.LayerType, self.input_size, self.num_classes)

    def forward_task(self, x, task_ids):
        # we recreate a prediction tensor of size [batch_size, self.global_num_classes]
        # we do so to get outputs of always same shape, the zeros should not interfere with prediction
        return torch.mul(self.forward(x), self.heads_mask[task_ids])


    def forward(self, x):
        x = x.view(-1, self.input_size)

        if self.method=='ogd':
            list_out = []
            for head in self.list_heads:
                list_out.append(head(x))
            x = torch.cat(list_out, dim=1)
        else:
            x = self.layer(x)
            assert x.shape[-1]==self.num_classes
        return x


    def get_loss(self, out, labels, loss_func, masked=False):

        if "MIMO_" in self.LayerType:
            loss=self.layer.get_loss(out, labels, loss_func, masked)
        else:
            if masked:
                classes_mask = torch.eye(self.num_classes).cuda()
                out = torch.mul(out, classes_mask[labels])
            loss = loss_func(out, labels)
        return loss