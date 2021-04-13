

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
        self.classifier = nn.Sequential(*list(model.children())[-1][:-1])
        self.features = nn.Sequential(*list(model.children())[:-1])


    def get_last_layer(self):
        return self.last

    def feature_extractor(self, x):
        x=self.features(x)
        x=x.view(-1, 9216)
        return self.classifier(x)

    def forward(self, x):
        x=x.view(-1,3,224,224)
        x=self.feature_extractor(x)
        return self.last(x)

    def update_head(self, batch, labels):
        # for SLDA

        batch = self.feature_extractor(batch)
        self.get_last_layer().update(batch, labels)

