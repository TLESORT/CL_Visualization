import torch
import torch.nn as nn
from torchvision import models
from Models.resnet import cifar_resnet20
from Models.model_utils import get_Output_layer, freeze_model



def CIFARModel(num_classes=10, OutLayer="Linear", pretrained_on=None, finetuning=False):

    if pretrained_on is not None:
        model = cifar_resnet20(pretrained=pretrained_on)
        model.num_classes = num_classes  # manual correction
    else:
        model = cifar_resnet20()
    latent_dim = model.fc.in_features

    if (pretrained_on is not None) and (not finetuning):
        model = freeze_model(model)

    model.fc = get_Output_layer(OutLayer, latent_dim, num_classes)
    return model
