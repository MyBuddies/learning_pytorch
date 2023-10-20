import pandas as pd
import numpy as np
import scipy.io

'''
比赛结果输出
'''
mat_data = scipy.io.loadmat(f'./data/testing/new_test_whole_data_6.mat')

# 提取Te工作区中的data和marker数据
if 'Te' in mat_data:
    Te = mat_data['Te']
    data = Te['data'][0, 0]
    marker = Te['marker'][0, 0]

    # 现在你可以使用data和marker进行后续处理
    # 例如，打印它们的形状
    print("data shape:", data.shape)
    print("marker shape:", marker.shape)
else:
    print("Te not found in the .mat file")

# for i in range(len(marker)):
#     if marker[i] == 14:
#         marker[i] = 1
#     elif marker[i] == 16:
#         marker[i] = 4
#     elif marker[i] == 17:
#         marker[i] = 2
#     elif marker[i] == 19:
#         marker[i] = 5

# for i in range(len(marker)):
#     if marker[i] == 15:
#         marker[i] = 2
#     elif marker[i] == 16:
#         marker[i] = 5
#     elif marker[i] == 17:
#         marker[i] = 3

# for i in range(len(marker)):
#     if marker[i] == 13:
#         marker[i] = 2
#     elif marker[i] == 17:
#         marker[i] = 4
#     elif marker[i] == 18:
#         marker[i] = 3
#     elif marker[i] == 19:
#         marker[i] = 5

# for i in range(len(marker)):
#     if marker[i] == 12:
#         marker[i] = 1
#     elif marker[i] == 16:
#         marker[i] = 2
#     elif marker[i] == 17:
#         marker[i] = 4
#     elif marker[i] == 18:
#         marker[i] = 3

for i in range(len(marker)):
    if marker[i] == 12:
        marker[i] = 1
    elif marker[i] == 14:
        marker[i] = 2
    elif marker[i] == 17:
        marker[i] = 4
    elif marker[i] == 20:
        marker[i] = 5

for i in range(len(marker)):
    if marker[i] == 12:
        marker[i] = 1
    elif marker[i] == 14:
        marker[i] = 2
    elif marker[i] == 16:
        marker[i] = 3
    elif marker[i] == 18:
        marker[i] = 4
    elif marker[i] == 20:
        marker[i] = 5

Te = {'data': data, 'marker': marker}
scipy.io.savemat('./data/testing/result_test_data_6.mat', {'Te': Te})


# # 创建一个包含数字的一维NumPy数组
# data = np.array([1, 2, 3, 4, 5])
#
# # 转换一维NumPy数组为DataFrame
# df = pd.DataFrame({'Column1': data})
#
# # 将DataFrame写入Excel文件
# excel_file = 'output.xlsx'  # 文件名
# df.to_excel(excel_file, index=False)
#
# print(f'数据已写入 {excel_file}')