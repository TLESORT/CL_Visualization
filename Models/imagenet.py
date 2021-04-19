

import torch
import torch.nn as nn
from torchvision import models

from Models.Output_Layers.head import NNHead

class ImageNetModel(nn.Module):

    def __init__(self, num_classes=10,
                 pretrained=False,
                 name_model="resnet",
                 OutLayer="Linear",
                 classes_per_head=None,
                 method="baseline"):
        super(ImageNetModel, self).__init__()
        self.pretrained = pretrained
        self.name_model = name_model
        self.num_classes = num_classes
        self.data_encoded = False
        if name_model=="alexnet":
            model = models.alexnet(pretrained=True)
        else:
            raise Exception("Ca va pas la")

        self.latent_dim  = list(model.children())[-1][-1].in_features

        self.head = NNHead(input_size=self.latent_dim,
                           num_classes=self.num_classes,
                           classes_per_tasks=classes_per_head,
                           LayerType=OutLayer,
                           method=method)

        self.classifier = nn.Sequential(*list(model.children())[-1][:-1])
        self.features = nn.Sequential(*list(model.children())[:-1])

    def set_data_encoded(self, flag):
        self.data_encoded = flag

    def get_last_layer(self):
        return self.head.layer

    def feature_extractor(self, x):
        x=self.features(x)
        x=x.view(-1, 9216)
        return self.classifier(x)

    def forward(self, x):
        if not self.data_encoded:
            x=x.view(-1,3,224,224)
            x=self.feature_extractor(x)
        return self.head(x)

    def update_head(self, batch, labels):
        # for SLDA

        if not self.data_encoded:
            batch = self.feature_extractor(batch)
        self.get_last_layer().update(batch, labels)

