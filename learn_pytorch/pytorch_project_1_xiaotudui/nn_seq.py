import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter


# class Model(nn.Module):
#     def __init__(self):
#         super(Model,self).__init__()
#         self.conv1 = Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2)
#         self.maxpool1 = MaxPool2d(kernel_size=2)
#         self.conv2 = Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
#         self.maxpool2 = MaxPool2d(kernel_size=2)
#         self.conv3 = Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
#         self.maxpool3 = MaxPool2d(kernel_size=2)
#         self.flatten = Flatten() # 64*4*4 = 1024
#         self.linear1 = Linear(in_features=1024, out_features=64)
#         self.linear2 = Linear(in_features=64, out_features=10)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.maxpool1(x)
#         x = self.conv2(x)
#         x = self.maxpool2(x)
#         x = self.conv3(x)
#         x = self.maxpool3(x)
#         x = self.flatten(x)
#         x = self.linear1(x)
#         x = self.linear2(x)
#         return x


# 引入 Sequential() 方法更简便
class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.model1 = Sequential(
            Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Flatten(),
            Linear(in_features=1024, out_features=64),
            Linear(in_features=64, out_features=10)
        )

    def forward(self, x):
        x = self.model1(x) # 简便之处，体现在这里
        return x


model = Model()
input = torch.ones(64, 3, 32, 32)
output = model(input)
print(output.shape)

writer = SummaryWriter("nn_seq")
writer.add_graph(model, input)
writer.close()