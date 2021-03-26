import torch.nn as nn
import torch
import numpy as np

from Models.model import Model

from Models.layer import CosineLayer

class MultiHead_Model(Model):
    def __init__(self, num_classes=10, classes_per_tasks=None, cosLayer=False):
        super(MultiHead_Model, self).__init__(num_classes)

        assert len(np.unique(classes_per_tasks))==num_classes

        self.num_head=len(classes_per_tasks)

        # vector that for each class gives the correct head index
        self.classes_heads=torch.zeros(num_classes)
        self.heads_mask=torch.zeros(self.num_head, num_classes).cuda()

        if cosLayer:
            last_layer_type=CosineLayer
        else:
            last_layer_type=nn.Linear

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
                self.list_heads.append(last_layer_type(50, dim).cuda())
            del self.last
            self.last = self.list_heads  # for ogd
        else:
            raise AssertionError("You should use a normal model if you do not define heads_dim")

        self.marginalized_class = None

    def forward_task(self, x, task_ids):
        # we recreate a prediction tensor of size [batch_size, self.global_num_classes]
        # we do so to get outputs of always same shape, the zeros should not interfere with prediction
        return torch.mul(self.forward(x), self.heads_mask[task_ids])


    def forward(self, x, latent_vector=False, head=None):
        x = x.view(-1, 1, 28, 28)
        x = self.feature_extractor(x)

        if not latent_vector:
            list_out = []
            for head in self.list_heads:
                list_out.append(head(x))
            x = torch.cat(list_out, dim=1)
        return x
