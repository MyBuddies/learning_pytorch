# import scipy.io
# import matplotlib.pyplot as plt
#
#
# mat_data = scipy.io.loadmat('./data/training/new_train_data_1.mat')
#
# # 提取Tr工作区中的data和marker数据
# if 'Tr' in mat_data:
#     Tr = mat_data['Tr']
#     data = Tr['data'][0, 0]
#     marker = Tr['marker'][0, 0]
# else:
#     print("Tr not found in the .mat file")
#
# # plt.imshow(data[1])
# # plt.show()
# # print(marker[1,0])
# # print(type(data[0]))
# # print(data[0].shape())
#
# train_data = []
# test_data = []
#
# train_data_label = []
# test_data_label = []
#
# print(len(marker))
#
# for i in range(len(marker)):
#     if i < 0.8*len(marker):
#         train_data.append(data[i])
#         train_data_label.append(marker[i,0])
#     else:
#         test_data.append(data[i])
#         test_data_label.append(marker[i,0])
#
# print(len(train_data))
# print(len(test_data))
# print(train_data_label[0])
# print(test_data_label[0])

import pandas as pd
import numpy as np

# 创建一个包含数字的一维NumPy数组
data = np.array([1, 2, 3, 4, 5])

# 转换一维NumPy数组为DataFrame
df = pd.DataFrame({'Column1': data})

# 将DataFrame写入Excel文件
excel_file = 'output.xlsx'  # 文件名
df.to_excel(excel_file, index=False)

print(f'数据已写入 {excel_file}')