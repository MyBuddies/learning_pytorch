import torch
import torchvision
import scipy.io
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import *
'''
模型训练
'''

# 定义训练的设备
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print("训练的设备是：{}".format(device))

for sub in range(6):
    # 准备数据集
    # train_data = torchvision.datasets.CIFAR10(root="./data_set", train=True, transform=torchvision.transforms.ToTensor(), download=True)
    # test_data = torchvision.datasets.CIFAR10(root="./data_set", train=False, transform=torchvision.transforms.ToTensor(), download=True)
    mat_data = scipy.io.loadmat(f'./data/training/new_train_data_{sub+1}.mat')

    # 提取Tr工作区中的data和marker数据
    if 'Tr' in mat_data:
        Tr = mat_data['Tr']
        data = Tr['data'][0, 0]
        marker = Tr['marker'][0, 0]
    else:
        print("Tr not found in the .mat file")

    train_data = []
    test_data = []

    train_data_label = []
    test_data_label = []

    for i in range(len(marker)):
        if i < 0.8*len(marker):
            train_data.append(torch.from_numpy(data[i]))
            train_data_label.append(torch.from_numpy(marker[i,0]-1))
        else:
            test_data.append(torch.from_numpy(data[i]))
            test_data_label.append(torch.from_numpy(marker[i,0]-1))


    # 数据集长度
    train_data_size = len(train_data)
    test_data_size = len(test_data)
    print("训练数据集长度：{}".format(train_data_size))
    print("测试数据集长度：{}".format(test_data_size))

    # 利用DataLoader加载数据
    # train_dataloader = DataLoader(train_data, batch_size=32)
    # test_dataloader = DataLoader(test_data, batch_size=32)

    # 创建网络模型
    model = Model()
    model = model.to(device) # 模型需要cuda

    # 损失函数
    loss_fn = nn.CrossEntropyLoss()
    loss_fn = loss_fn.to(device) # 损失函数需要cuda

    # 优化器
    learning_rate = 5e-5
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 设置一些网络参数
    # 记录训练的次数
    total_train_step = 0
    # 记录测试的次数
    total_test_step = 0
    # 训练的轮数
    epoch = 25

    # 添加tensorboard
    writer = SummaryWriter("logs_train")

    for e in range(epoch):
        print(f"------subject{sub+1}的第{e+1}轮训练开始------")

        # 训练步骤开始
        model.train()

        for i in range(len(train_data)):
            train_data[i] = train_data[i].to(device) # 数据需要cuda
            train_data_label[i] = train_data_label[i].to(device) # 数据需要cuda
            # image = torch.reshape(image,(1,3,32,32))
            train_data[i] = torch.reshape(train_data[i], (1,1,200,22))
            output = model(train_data[i].float())
            loss = loss_fn(output, train_data_label[i])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_step += 1
            if total_train_step % 50 == 0:
                print("训练次数：{}，Loss：{}".format(total_train_step, loss.item()))
                # writer.add_scalar("train_loss", loss.item(), total_train_step)


        # 测试步骤开始
        model.eval()

        total_test_loss = 0
        total_accuracy = 0
        with torch.no_grad(): # 测试不用梯度
            for i in range(len(test_data)):
                test_data[i] = test_data[i].to(device) # 数据需要cuda
                test_data_label[i] = test_data_label[i].to(device) # 数据需要cuda
                test_data[i] = torch.reshape(test_data[i], (1,1,200,22))
                output = model(test_data[i].float())
                loss = loss_fn(output.float(), test_data_label[i])
                total_test_loss += loss.item()
                accuacy = (output.argmax(1) == test_data_label[i]).sum()
                total_accuracy += accuacy
        print("整体测试集上的Loss：{}".format(total_test_loss))
        print("整体测试集上的Accuracy：{}".format(total_accuracy/test_data_size))
        # writer.add_scalar("test_loss", total_test_loss, total_test_step)
        # writer.add_scalar("test_Accuracy", total_accuracy/test_data_size, total_test_step)
        total_test_step += 1

        torch.save(model, f"./pretrained_model_{sub+1}/model{e+1}.pth")
        print("模型已保存")

    writer.close()