import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import *

# 定义训练的设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("训练的设备是：{}".format(device))

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="./data_set", train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10(root="./data_set", train=False, transform=torchvision.transforms.ToTensor(), download=True)

# 数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集长度：{}".format(train_data_size))
print("测试数据集长度：{}".format(test_data_size))

# 利用DataLoader加载数据
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
model = Model()
model = model.to(device) # 模型需要cuda

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device) # 损失函数需要cuda

# 优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 设置一些网络参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 50

# 添加tensorboard
writer = SummaryWriter("logs_train")

for i in range(epoch):
    print("------第{}轮训练开始------".format(i+1))

    # 训练步骤开始
    model.train()

    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device) # 数据需要cuda
        targets = targets.to(device) # 数据需要cuda
        output = model(imgs)
        loss = loss_fn(output, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数：{}，Loss：{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)


    # 测试步骤开始
    model.eval()

    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad(): # 测试不用梯度
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device) # 数据需要cuda
            targets = targets.to(device) # 数据需要cuda
            output = model(imgs)
            loss = loss_fn(output, targets)
            total_test_loss += loss.item()
            accuacy = (output.argmax(1) == targets).sum()
            total_accuracy += accuacy
    print("整体测试机上的Loss：{}".format(total_test_loss))
    print("整体测试机上的Accuracy：{}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_Accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step += 1

    torch.save(model, "./pretrained_model/model{}.pth".format(i+1))
    print("模型已保存")

writer.close()