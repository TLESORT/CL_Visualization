
import torch.nn as nn
import torch

class Model(nn.Module):
    def __init__(self, num_classes=10):
        super(Model, self).__init__()
        self.global_num_classes = num_classes

        self.input_dim = 1
        self.output_dim = 1
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, self.global_num_classes)
        self.marginalized_class = None

    def set_marginalized_class(self, marginalized_class):
        assert marginalized_class < self.global_num_classes
        self.marginalized_class = marginalized_class

    def forward(self, x, latent_vector=False):
        x = x.view(-1, 1, 28, 28)

        x = self.relu(self.maxpool2(self.conv1(x)))
        x = self.relu(self.maxpool2(self.conv2(x)))
        x = x.view(-1, 320)
        x = self.relu(self.fc1(x))
        if not latent_vector:
            x = self.fc2(x)
        if self.marginalized_class is not None:
            x_class = x[:, self.marginalized_class]
            x_others = torch.cat([x[:,:self.marginalized_class],x[:,self.marginalized_class+1:]], axis=1)
            x_others = torch.max(x_others, axis=1)
            x = torch.cat([x_class, x_others], axis=1)
            assert x.shape[1] == 2
        return x
