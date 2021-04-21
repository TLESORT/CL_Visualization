import torch.nn as nn
import torch

from Models.Output_Layers.head import NNHead


class Model(nn.Module):
    def __init__(self, num_classes=10, OutLayer="Linear", classes_per_head=None, method="baseline", pretrained_on=None):
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
                           classes_per_tasks=classes_per_head,
                           LayerType=OutLayer,
                           method=method)

    def get_last_layer(self):
        return self.head.layer

    def feature_extractor(self, x):
        x = self.relu(self.maxpool2(self.conv1(x)))
        x = self.relu(self.maxpool2(self.conv2(x)))
        x = x.view(-1, 320)
        return self.linear(x)

    def forward_task(self, x, ind_task):
        x = x.view(-1, 1, 28, 28)
        x = self.feature_extractor(x)
        x = self.head.forward_task(x, ind_task)
        return x

    def forward(self, x, latent_vector=False):
        x = x.view(-1, 1, 28, 28)
        x = self.feature_extractor(x)
        if not latent_vector:
            x = self.head(x)

        return x

    def get_loss(self, out, labels, loss_func):
        return self.get_last_layer().get_loss(out, labels, loss_func)