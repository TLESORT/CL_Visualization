import torch
import torch.nn as nn

import numpy as np

from Models.Output_Layers.StreamingSLDA import StreamingLDA
from Models.model_utils import get_Output_layer

import math


class CosineLayer(nn.Module):
    def __init__(self, size_in, size_out):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        weight = torch.Tensor(size_out, size_in)
        self.weight = nn.Parameter(weight)  # nn.Parameter is a Tensor that's a module parameter.

        # initialize weights
        nn.init.kaiming_normal_(self.weight)  # weight init

    def forward(self, x):
        cosine_out = []
        cosine_out
        for i in range(self.size_out):
            cosine_out.append(torch.cosine_similarity(x, self.weight[i, :].unsqueeze(0)).unsqueeze(-1))

        x = torch.cat(cosine_out, dim=1)
        return x



class MIMO(nn.Module):
    def __init__(self, size_in, size_out, num_layer=3, layer_type="MIMO_Linear"):
        super().__init__()
        self.num_layer = num_layer
        self.layer_type = layer_type.replace("MIMO_", "")
        self.size_in, self.size_out = size_in, size_out
        self.classes_mask = torch.eye(self.size_out).cuda()

        self.layer = get_Output_layer(self.layer_type, self.size_in, self.size_out * self.num_layer).cuda()

    def forward(self, x):
        x = self.layer(x)
        x = x.view(-1, self.num_layer, self.size_out)
        return x

    def get_loss(self, out, labels, loss_func, masked=False):
        # out: batch, num_layer, self.size_out
        loss = 0
        # for each layer we compute the loss and we sum
        for i in range(self.num_layer):
            if masked:
                masked_out = torch.mul(out[:, i, :], self.classes_mask[labels])
                loss += loss_func(masked_out, labels)
            else:
                loss += loss_func(out[:, i, :], labels)
        return loss


# Nearest Prototype (Similar to ICARL)
class MeanLayer(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """

    def __init__(self, size_in, size_out):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        self.data = torch.zeros(0, size_in)
        self.labels = torch.zeros(0)
        self.weight = torch.zeros(size_out, size_in) # mean layer
        self.nb_inst = torch.zeros(size_out)
        self.initiated = False

    def forward(self, x):
        data = x.detach().cpu()  # no backprop possible

        assert not torch.isnan(data).any()

        if self.initiated:
            #torch.cdist(c * b, d * b) -> c*d
            out = torch.cdist(data, self.weight)
            # convert smaller is better into bigger in better
            out = out * -1
        else:
            # if mean are not initiate we return random predition
            out = torch.randn((data.shape[0], self.size_out)).cuda()
        return out.cuda()

    def update(self, epoch=0):
        pass

    def accumulate(self, x, y, epoch=0):
        if epoch == 0:
            self.data = x.view(-1, self.size_in).detach().cpu()
            self.labels = y
            for i in range(self.size_out):
                indexes = torch.where(self.labels == i)[0]
                self.weight[i] = (self.weight[i] * (1.0 * self.nb_inst[i]) + self.data[indexes].sum(0))
                self.nb_inst[i] += len(indexes)
                if self.nb_inst[i] != 0:
                    self.weight[i] = self.weight[i] / (1.0 * self.nb_inst[i])

            self.data = torch.zeros(0, self.size_in)
            self.labels = torch.zeros(0)
            self.initiated = True

        assert not torch.isnan(self.weight).any()


class MedianLayer(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """

    def __init__(self, size_in, size_out):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        self.data = torch.zeros(0, size_in)
        self.labels = torch.zeros(0)
        self.weight = torch.zeros(size_out, size_in) # median vector
        self.initiated = False

    def forward(self, x):
        data = x.detach().cpu()  # no backprop possible

        assert not torch.isnan(data).any()

        if self.initiated:
            out = torch.cdist(data, self.weight)
            # convert smaller is better into bigger in better
            out = out * -1
        else:
            # if mean are not initiate we return random predition
            out = torch.randn((data.shape[0], self.size_out))
        return out.cuda()

    def update(self, epoch=0):
        if epoch == 0:
            for i in range(self.size_out):
                indexes = torch.where(self.labels == i)[0]
                if len(indexes)>0:
                    self.weight[i], _ = torch.median(self.data[indexes], dim=0)
            self.initiated = True

    def accumulate(self, x, y, epoch=0):
        if epoch == 0:
            x = x.view(-1, self.size_in)
            self.data = torch.cat([self.data, x.cpu() .detach()])
            self.labels = torch.cat([self.labels, y.cpu() ])


class SLDALayer(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """

    def __init__(self, size_in, size_out):
        super().__init__()
        self.size_in = size_in
        self.size_out = size_out
        self.slda = StreamingLDA(input_shape=size_in, num_classes=size_out, test_batch_size=1024, shrinkage_param=1e-4,
                                 streaming_update_sigma=True).cuda()
        self.initiated = False
        self.data = torch.zeros((0, size_in)).cuda()
        self.labels = torch.zeros(0).cuda()

    def forward(self, x):
        if self.initiated:
            x = self.slda.predict(x)
        else:
            x = torch.randn((x.shape[0], self.size_out)).cuda()

        return x.cuda()




    def accumulate(self, x, y, epoch=0):

        if epoch == 0:
            self.initiated = True
            x = x.view(-1, self.size_in)
            for i in range(len(y)):
                self.slda.fit(x[i], y[i])


    def update(self, epoch=0):
        pass



class KNN(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """

    def __init__(self, size_in, size_out, K=5):
        super().__init__()
        from sklearn.neighbors import KNeighborsClassifier
        self.neigh = KNeighborsClassifier(n_neighbors=K, weights='distance', algorithm='brute')
        self.data = torch.zeros((0, size_in))
        self.labels = torch.zeros(0)
        self.size_out = size_out
        self.size_in = size_in
        self.initiated = False
        self.classes_mask = torch.eye(self.size_out).cuda()

    def forward(self, x):
        data = x.detach()  # no backprop possible
        if self.initiated:
            classes = self.neigh.predict(data.cpu().numpy())
            out = self.classes_mask[classes]
        else:
            out = torch.randn((x.shape[0], self.size_out)).cuda()
        return out


    def _trim_data(self):
        new_data = []
        new_labels = []

        for i in range(self.size_out):
            indexes = torch.where(self.labels == i)[0]
            data_classes = self.data[indexes]
            if len(indexes) > 250:
                dists = torch.cdist(data_classes, data_classes).sum(axis=0)
                _, sort_index = dists.sort()
                # we keep 150 samples
                indexes2keep = indexes[sort_index[:150]]
                new_data.append(self.data[indexes2keep])
                new_labels.append(self.labels[indexes2keep])
            elif len(indexes) > 0:
                new_data.append(self.data[indexes])
                new_labels.append(self.labels[indexes])

        if len(new_labels)>0:
            self.data = torch.cat(new_data, axis=0)
            self.labels = torch.cat(new_labels, axis=0)

    def accumulate(self, x, y, epoch=0):

        if epoch == 0:
            x = x.view(-1, self.size_in)
            self.data = torch.cat([self.data, x.detach().cpu()], axis=0)
            self.labels = torch.cat([self.labels, y.cpu()], axis=0)

            if len(self.labels) > 200 * self.size_out:
                self._trim_data()


    def update(self, epoch=0):
        if epoch == 0:
            self._trim_data()
            self.neigh.fit(self.data.numpy(), self.labels.numpy())
            self.initiated = True
