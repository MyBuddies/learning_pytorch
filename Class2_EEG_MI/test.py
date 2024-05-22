import torchvision.transforms
from PIL import Image
from model import *
import scipy.io
import pandas as pd

'''
模型测试
'''

device = torch.device('cpu')

for sub in range(6):
    mat_data = scipy.io.loadmat(f'./data/testing/new_test_data_{sub+1}.mat')

    # 提取Tr工作区中的data和marker数据
    if 'Te' in mat_data:
        Te = mat_data['Te']
        data = Te['data'][0, 0]
        marker = Te['marker'][0, 0]
    else:
        print("Te not found in the .mat file")

    test_data = []

    for i in range(len(marker)):
        test_data.append(torch.from_numpy(data[i]))


    model =torch.load(f"./pretrained_model/pretrained_model_{sub+1}/model25.pth")
    # print(model)

    result = []
    model.eval()
    with torch.no_grad():
        for i in range(len(marker)):
            test_data[i] = torch.reshape(test_data[i], (1, 1, 200, 22))
            output = model(test_data[i].float())
            # print(output.argmax(1))
            result.append(int(output.argmax(1))+1)

    df = pd.DataFrame(result)

    # 将DataFrame写入Excel文件
    excel_file = f'./results/output_{sub+1}.xlsx'  # 文件名
    df.to_excel(excel_file, index=False)

    print(f'数据已写入 {excel_file}')

    # output = model(image.to(device))

# print(output.argmax(1))