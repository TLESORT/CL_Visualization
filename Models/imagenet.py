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
        self.img_size = 224

        if self.name_model == "alexnet":
            model = models.alexnet(pretrained=True)
            self.latent_dim = list(model.children())[-1][-1].in_features #2048
            self.classifier = nn.Sequential(*list(model.children())[-1][:-1])  # between features and outlayer
            self.features = nn.Sequential(*list(model.children())[:-1])
            self.features_size = 9216
        elif self.name_model == "resnet":
            model = models.resnet18(pretrained=True)
            self.latent_dim = list(model.children())[-1].in_features  #512
            self.features = nn.Sequential(*list(model.children())[:-1])
        elif self.name_model == "googlenet":
            model = models.googlenet(pretrained=True)
            self.latent_dim = list(model.children())[-1].in_features # 1024
            self.features = nn.Sequential(*list(model.children())[:-1])
        elif self.name_model == "vgg":
            model = models.vgg16(pretrained=True)
            self.latent_dim = list(model.children())[-1][-1].in_features #2048
            self.classifier = nn.Sequential(*list(model.children())[-1][:-1])
            self.features = nn.Sequential(*list(model.children())[:-1])
            self.features_size = 25088
        else:
            raise Exception("Ca va pas la")



        self.head = NNHead(input_size=self.latent_dim,
                           num_classes=self.num_classes,
                           classes_per_tasks=classes_per_head,
                           LayerType=OutLayer,
                           method=method)


    def set_data_encoded(self, flag):
        self.data_encoded = flag
        if self.data_encoded:
            # we can free some memory if the data is already encoded
            self.classifier = None
            self.features = None

    def get_last_layer(self):
        return self.head.layer

    def feature_extractor(self, x):

        if self.name_model in ["alexnet", "vgg"]:
            x = self.classifier(self.features(x).view(-1, self.features_size))
        else:
            x = self.features(x)
        return x

    def forward(self, x):
        if not self.data_encoded:
            x = x.view(-1, 3, self.img_size, self.img_size)
            x = self.feature_extractor(x)
        x = x.view(-1, self.latent_dim)
        return self.head(x)

    def accumulate(self, batch, labels, epoch=0):

        if not self.data_encoded:
            batch = self.feature_extractor(batch)

        batch = batch.view(batch.size(0), -1)
        self.get_last_layer().accumulate(batch, labels, epoch)

    def update_head(self, epoch):
        self.get_last_layer().update(epoch)

    def get_loss(self, out, labels, loss_func, masked=False):
        return self.head.get_loss(out, labels, loss_func, masked)
