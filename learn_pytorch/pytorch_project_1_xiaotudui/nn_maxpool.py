import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root="./data_set", train=False, transform=torchvision.transforms.ToTensor(), download=True)

dataloader = DataLoader(dataset, batch_size=64)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self, x):
        x = self.maxpool1(x)
        return x

model = Model()

writer = SummaryWriter("nn_maxpool")

step = 0
for data in dataloader:
    imgs, targets = data
    output = model(imgs)
    # print(output.shape)
    writer.add_images("input", imgs, step)
    # output = torch.reshape(output, (-1,3,30,30))
    writer.add_images("output", output, step)
    step += 1

writer.close()