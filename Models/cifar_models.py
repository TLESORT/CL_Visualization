import torch
import torch.nn as nn
from torchvision import models
from Models.resnet import cifar_resnet20
from Models.model_utils import get_Output_layer, freeze_model


class CIFARModel(nn.Module):

    def __init__(self, num_classes=10, OutLayer="Linear", pretrained_on=None, finetuning=False):
        super(CIFARModel, self).__init__()
        self.pretrained = pretrained_on
        self.num_classes = num_classes
        self.finetuning = finetuning

        if pretrained_on is not None:
            model = cifar_resnet20(pretrained=pretrained_on)
            model.num_classes = num_classes  # manual correction
        else:
            model = cifar_resnet20()
        latent_dim = model.fc.in_features

        self.features = nn.Sequential(*list(model.children())[:-1])
        if (pretrained_on is not None) and (not finetuning):
            self.features=freeze_model(self.features)
        self.last = get_Output_layer(OutLayer, latent_dim, num_classes)


    def get_last_layer(self):
        return self.last

    def feature_extractor(self, x):
        return self.features(x)

    def forward(self, x):
        # x=x.view(-1,3,32,32)
        x=self.feature_extractor(x)
        x=x.view(-1, 64)
        return self.last(x)

    def update_head(self, batch, labels):
        # for SLDA

        batch = self.feature_extractor(batch)
        self.get_last_layer().update(batch, labels)
