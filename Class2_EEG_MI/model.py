import torch
from torch import nn
from torch.nn import Sequential

'''
搭建卷积神经网络
'''
class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.model = Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=2),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=2),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=2),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(in_features=2688, out_features=64),
            # nn.ReLU(),
            nn.Linear(in_features=64, out_features=5)
        )

    def forward(self, x):
        x = self.model(x) # 简便之处，体现在这里
        return x

# 测试模型的正确性
if __name__ == "__main__":
    model = Model()
    print(model)
    input = torch.ones(1, 1, 200, 22)
    output = model(input)
    print(output.shape)