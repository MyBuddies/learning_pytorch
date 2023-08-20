import torch
import torchvision
from torch import nn
from torch.nn import ReLU, Sigmoid, Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root="./data_set", train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=64)


class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.liner1 = Linear(196608,10)

    def forward(self, input):
        output = self.liner1(input)
        return output


model = Model()

# writer = SummaryWriter("nn_sigmoid")

step = 0
for data in dataloader:
    imgs, targets = data

    print(imgs.shape)
    # writer.add_images("input", imgs, step)
    # output = torch.reshape(imgs, (1,1,1,-1))
    output = torch.flatten(imgs)
    print(output.shape)
    output = model(output)
    print(output.shape)
    # writer.add_images("output", output, step)
    step += 1

# writer.close()
