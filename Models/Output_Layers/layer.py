
import torch
import torch.nn as nn

import math
class CosineLayer(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self, size_in, size_out):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        weights = torch.Tensor(size_out, size_in)
        self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.

        #self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        # initialize weights and biases
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5)) # weight init
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)

    def forward(self, x):

        cosine_out = []
        for i in range(self.size_out):
            #cosine_out.append(self.cos(x, self.weights[i,:].unsqueeze(0)).unsqueeze(-1))
            cosine_out.append(torch.cosine_similarity(x, self.weights[i,:].unsqueeze(0)).unsqueeze(-1))

        x = torch.cat(cosine_out, dim=1)
        return x

class MultiHead(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self, size_in, size_out, head_sizes):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        weights = torch.Tensor(size_out, size_in)
        self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.

        #self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        # initialize weights and biases
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5)) # weight init
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)

    def forward(self, x):

        cosine_out = []
        for i in range(self.size_out):
            #cosine_out.append(self.cos(x, self.weights[i,:].unsqueeze(0)).unsqueeze(-1))
            cosine_out.append(torch.cosine_similarity(x, self.weights[i,:].unsqueeze(0)).unsqueeze(-1))

        x = torch.cat(cosine_out, dim=1)
        return x