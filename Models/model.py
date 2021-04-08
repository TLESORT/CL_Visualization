import torch.nn as nn
import torch

from Models.Output_Layers.head import NNHead


class Model(nn.Module):
    def __init__(self, num_classes=10, OutLayer="Linear"):
        super(Model, self).__init__()
        self.num_classes = num_classes

        self.input_dim = 1
        self.output_dim = 1
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(320, 50)

        self.linear = nn.Sequential(self.fc1,
                                    self.relu,
                                    )  # for ogd
        self.head = NNHead(input_size=50,
                           num_classes=self.num_classes,
                           classes_per_tasks=None,
                           LayerType=OutLayer,
                           method='baseline')


    def feature_extractor(self, x):
        x = self.relu(self.maxpool2(self.conv1(x)))
        x = self.relu(self.maxpool2(self.conv2(x)))
        x = x.view(-1, 320)
        return self.linear(x)

    def forward_task(self, x, ind_task):
        #todo
        pass

    def forward(self, x, latent_vector=False):
        x = x.view(-1, 1, 28, 28)
        x = self.feature_extractor(x)
        if not latent_vector:
            x = self.last(x)

        return x
