from torch.utils.data import DataLoader

from continuum.datasets import CIFAR10
from continuum import ClassIncremental

from torchvision import transforms

transformations=transforms.Compose([
    transforms.RandomCrop(size=32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465], # mean=[0.5071, 0.4865, 0.4409] for cifar100
        std=[0.2023, 0.1994, 0.2010], # std=[0.2009, 0.1984, 0.2023] for cifar100
    ),
])
dataset = CIFAR10("../Datasets", download=True, train=True)
scenario_tr = ClassIncremental(dataset, nb_tasks=1, transformations=[transformations])
#net = get_model('cifar_resnet110_v1', classes=10, pretrained=True)


from Models.resnet import cifar_resnet20

net = cifar_resnet20(pretrained="cifar10")

correct = 0
for task_set in scenario_tr:
    loader = DataLoader(task_set, batch_size=64, shuffle=True, num_workers=6)
    for x,y,i in loader:
        output=net(x)
        pred = output.max(dim=1)[1].cpu()
        correct += (pred == y).sum()

#pred = net(img.expand_dims(axis=0))