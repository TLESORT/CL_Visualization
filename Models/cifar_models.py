import torch
import torch.nn as nn
from torchvision import models
from Models.resnet import cifar_resnet20
from Models.model_utils import get_Output_layer, freeze_model

from Models.Output_Layers.head import NNHead


def CIFARModel(num_classes=10, OutLayer="Linear", classes_per_head=None, method="baseline", pretrained_on=None,
               model_dir=None, finetuning=False):

    if pretrained_on is not None:
        model = cifar_resnet20(pretrained=pretrained_on, model_dir=model_dir)
        model.num_classes = num_classes  # manual correction
    else:
        model = cifar_resnet20(num_classes=num_classes)
        model.num_classes = num_classes  # manual correction
    latent_dim = model.fc.in_features

    if (pretrained_on is not None) and (not finetuning):
        model = freeze_model(model)

    # model.fc = get_Output_layer(OutLayer, latent_dim, num_classes)
    model.head = NNHead(input_size=latent_dim,
                        num_classes=num_classes,
                        classes_per_tasks=classes_per_head,
                        LayerType=OutLayer,
                        method=method)
    return model
