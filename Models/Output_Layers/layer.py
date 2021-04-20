import torch
import torch.nn as nn
from Models.Output_Layers.StreamingSLDA import StreamingLDA
from Models.model_utils import get_Output_layer

import math


class CosineLayer(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """

    def __init__(self, size_in, size_out):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        weight = torch.Tensor(size_out, size_in)
        self.weight = nn.Parameter(weight)  # nn.Parameter is a Tensor that's a module parameter.

        # self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        # initialize weights and biases
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # weight init
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        self.bias = torch.zeros(size_out)

    def forward(self, x):
        cosine_out = []
        for i in range(self.size_out):
            cosine_out.append(torch.cosine_similarity(x, self.weight[i, :].unsqueeze(0)).unsqueeze(-1))

        x = torch.cat(cosine_out, dim=1)
        return x


class SLDALayer(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """

    def __init__(self, size_in, size_out):
        super().__init__()
        self.slda = StreamingLDA(input_shape=size_in, num_classes=size_out, test_batch_size=1024, shrinkage_param=1e-4,
                                 streaming_update_sigma=True).cuda()

    def forward(self, x):
        x = self.slda.predict(x)

        return x.cuda()

    def update(self, batch, labels):
        for i in range(len(labels)):
            self.slda.fit(batch[i], labels[i])


class MIMO(nn.Module):
    def __init__(self, size_in, size_out, num_layer=3, layer_type="Linear"):
        super().__init__()
        self.num_layer=num_layer
        self.size_in, self.size_out = size_in, size_out
        weight = torch.Tensor(size_out, size_in)
        self.weight = nn.Parameter(weight)  # nn.Parameter is a Tensor that's a module parameter.

        # create output layers
        self.list_layers = []
        for i in range(self.num_layer):
            self.list_layers.append(get_Output_layer(layer_type, self.size_in, self.size_out))


        # initialize weights and biases
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # weight init
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        self.bias = torch.zeros(size_out)

    def forward(self, x):
        list_out =[]
        for i in range(self.num_layer):
            list_out.append(self.list_layers[i](x))
        out=torch.cat(list_out, dim=1)

        return out.mean(1)

# Nearest Prototype (Similar to ICARL)
class MeanLayer(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """

    def __init__(self, size_in, size_out):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        self.data = torch.zeros(0, size_in).cuda()
        self.labels = torch.zeros(0).cuda()
        self.mean = torch.zeros(size_out, size_in).cuda()
        self.weight = torch.zeros(size_out)

    def forward(self, x):

        x = torch.cdist(x, self.mean)
        #convert smaller is better into bigger in better
        x = x * -1
        return x

    # def accumulate(self, x, y):
    #     x=x.view(-1, 64)
    #     self.data = torch.cat([self.data, x])
    #     self.labels = torch.cat([self.labels, y])

    def update(self, x, y):
        #self.accumulate(x, y)
        self.data = x.view(-1, 64)
        self.labels = y
        for i in range(self.size_out):
            indexes = torch.where(self.labels==i)[0]
            self.mean[i]=(self.mean[i] * (1.0 * self.weight[i]) + self.data[indexes].sum(0))
            self.weight[i] += len(indexes)
            if self.weight[i]!=0:
                self.mean[i] =self.mean[i] / (1.0 * self.weight[i])

            # remove accounted latent vector
            #indexes2keep = torch.where(self.labels!=i)
            #self.data = self.data[indexes2keep]
        self.data = torch.zeros(0, self.size_in).cuda()
        self.labels = torch.zeros(0).cuda()


class KNN(object):
    """ Custom Linear layer but mimics a standard linear layer """

    def __init__(self, size_in, size_out, K=5):
        super().__init__()
        from sklearn.neighbors import KNeighborsClassifier
        self.neigh = KNeighborsClassifier(n_neighbors=K)
        self.data = torch.zeros(0, size_in)
        self.labels = torch.zeros(0)

    def forward(self, x):
        x = self.neigh.predict(x)

        return x.cuda()

    def accumulate(self, x, y):
        self.data = torch.concatenate((self.data, x))
        self.labels = torch.concatenate((self.labels, y))

    def update(self):
        self.neigh.fit(self.data, self.labels)
