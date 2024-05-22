import torch
from  torch import nn

class Model(nn.Module):
    def __int__(self):
        super().__int__()

    def forward(self, input):
        output = input + 1
        return output


model = Model()
x = torch.tensor(1.0)
output = model(x)
print(output)