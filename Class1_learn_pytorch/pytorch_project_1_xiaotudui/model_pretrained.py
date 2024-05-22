import torchvision
from torch import nn

# 对现有网络模型的使用与修改
vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)

train_data = torchvision.datasets.CIFAR10(root="./data_set", train=False, transform=torchvision.transforms.ToTensor(), download=True)

# 加层
vgg16_true.classifier.add_module("add_linear", nn.Linear(1000, 10))
print(vgg16_true)

# 修改线形层
vgg16_false.classifier[6] = nn.Linear(4096, 10)
print(vgg16_false)