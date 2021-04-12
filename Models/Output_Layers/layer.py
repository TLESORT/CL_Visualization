
import torch
import torch.nn as nn
from Models.Output_Layers.StreamingSLDA import StreamingLDA

import math
class CosineLayer(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self, size_in, size_out):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        weight = torch.Tensor(size_out, size_in)
        self.weight = nn.Parameter(weight)  # nn.Parameter is a Tensor that's a module parameter.

        #self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        # initialize weights and biases
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5)) # weight init
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        self.bias = torch.zeros(size_out)

    def forward(self, x):

        cosine_out = []
        for i in range(self.size_out):
            cosine_out.append(torch.cosine_similarity(x, self.weight[i,:].unsqueeze(0)).unsqueeze(-1))

        x = torch.cat(cosine_out, dim=1)
        return x

class SLDALayer(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self, size_in, size_out):
        super().__init__()
        self.slda = StreamingLDA(input_shape=size_in, num_classes=size_out, test_batch_size=1024, shrinkage_param=1e-4,
                 streaming_update_sigma=True).cuda()

    def forward(self, x):
        x=self.slda.predict(x)

        return x.cuda()

    def update(self, batch, labels):

        for i in range(len(labels)):
            self.slda.fit(batch[i],labels[i])

class KNN(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self, size_in, size_out, K=5):
        super().__init__()
        from sklearn.neighbors import KNeighborsClassifier
        self.neigh = KNeighborsClassifier(n_neighbors=K)
        self.data = torch.zeros(0, size_in)
        self.labels = torch.zeros(0)

    def forward(self, x):
        x=self.neigh.predict(x)

        return x.cuda()

    def accumulate(self, x ,y):

        self.data = torch.concatenate((self.data, x))
        self.labels = torch.concatenate((self.labels, y))

    def update(self):
        self.neigh.fit(self.data, self.labels)