

import torch
import torch.nn as nn
from torchvision import models


class ImageNetModel(nn.Module):

    def __init__(self, num_classes=10, pretrained=False, name_model="resnet"):
        super(ImageNetModel, self).__init__()
        self.pretrained = pretrained
        self.name_model = name_model
        self.num_classes = num_classes
        if name_model=="alexnet":
            model = models.alexnet(pretrained=True)
        else:
            raise Exception("Ca va pas la")

        self.last = nn.Linear(list(model.children())[-1][-1].in_features, num_classes)
        block = nn.Sequential(*list(model.children())[-1][:-1])

        self.features = nn.Sequential(*list(model.children())[:-1], block)


    def get_last_layer(self):
        return self.last

    def feature_extractor(self, x):
        return self.features(x)

    def forward(self, x):
        self.feature_extractor(x)
        return self.last(x)

    def update_head(self, batch, labels):
        # for SLDA

        batch = self.feature_extractor(batch)
        self.get_last_layer().update(batch, labels)

