import torch.nn as nn
import torch

from Models.model import Model


class ClassMarginalization(Model):
    def __init__(self, num_classes=10, OutLayer="Linear", marginalized_class=0):
        super(ClassMarginalization, self).__init__(num_classes, OutLayer)

        assert marginalized_class < self.global_num_classes
        self.marginalized_class = marginalized_class

    def feature_extractor(self, x):
        x = self.relu(self.maxpool2(self.conv1(x)))
        x = self.relu(self.maxpool2(self.conv2(x)))
        x = x.view(-1, 320)
        return self.linear(x)

    def forward(self, x, latent_vector=False):
        assert latent_vector==False, print("Class marginalization is not compatible with latent vector = True")
        x = super().forward(x)

        # TODO: test this
        # Marginalization: We convert the classifier into a binary classifier class to marginalized vs others
        x_class = x[:, self.marginalized_class]
        x_others = torch.cat([x[:, :self.marginalized_class], x[:, self.marginalized_class + 1:]], axis=1)
        x_others = torch.max(x_others, axis=1)
        x = torch.cat([x_class, x_others], axis=1)
        assert x.shape[1] == 2
        return x
