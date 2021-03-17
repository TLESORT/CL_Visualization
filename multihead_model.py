import torch.nn as nn
import torch

from model import Model


class MultiHead_Model(Model):
    def __init__(self, num_classes=10, heads_dim=None):
        super(MultiHead_Model, self).__init__(num_classes)

        self.multi_heads = False

        if heads_dim is not None:
            self.multi_heads = True
            self.list_heads = []
            # we suppose no overlaps in heads
            for i, dim in enumerate(heads_dim):
                self.list_heads.append(nn.Linear(50, dim))
            del self.last
            self.last = self.list_heads  # for ogd
        else:
            raise AssertionError("You should use a normal model if you do not define heads_dim")

        self.marginalized_class = None

    def forward_task(self, x, task_ids):
        x = x.view(-1, 1, 28, 28)
        x = self.feature_extractor(x)

        # we recreate a prediction tensor of size [batch_size, self.global_num_classes]
        # we do so to get outputs of always same shape, the zeros should not interfere with prediction
        predictions = torch.zeros(x.shape[0], self.global_num_classes).cuda()

        # todo find a way to be able to make batch inference
        if x.shape[0] > 1:
            for i, (x_, t_) in enumerate(zip(x, task_ids)):
                index_head = t_ // 2  # todo fix this so be compatible with other multi head settings
                predictions[i, index_head * 2:(index_head + 1) * 2] = self.list_heads[index_head].cuda()(x_)
        else:
            index_head = task_ids.item() // 2
            predictions[0, index_head * 2:(index_head + 1) * 2] = self.list_heads[index_head].cuda()(x[0])

        # we recreate the batch so dim 0
        return predictions

    def forward(self, x, latent_vector=False):
        x = x.view(-1, 1, 28, 28)
        x = self.feature_extractor(x)

        if not latent_vector:
            list_out = []
            for head in self.list_heads:
                list_out.append(head(x))
            x = torch.cat(list_out, dim=1)

        return x
