import scipy.io

'''
# 训练数据集数据预处理
'''

for n in range(6):
    # 读取.mat文件
    mat_data = scipy.io.loadmat(f'./data/training/train_subject_{n+1}.mat')

    # 提取Tr工作区中的data和marker数据
    if 'Tr' in mat_data:
        Tr = mat_data['Tr']
        data = Tr['data'][0, 0]
        marker = Tr['marker'][0, 0]

        # 现在你可以使用data和marker进行后续处理
        # 例如，打印它们的形状
        print("data shape:", data.shape)
        print("marker shape:", marker.shape)
    else:
        print("Tr not found in the .mat file")


    # print(marker[71257])
    # print(data[71257])
    newData = []
    newMarker = [] 
    
    # for i in range(len(marker)):
    #     if marker[i] in [1,2,3,4,5]:
    #         newData.append(data[i])
    #         newMarker.append(marker[i])
    
    # i = 0
    # flag = -1
    # while i < len(marker):
    #     if marker[i] != flag:
    #             newData.append(data[i:i+200,:])
    #             newMarker.append(marker[i:i+200])
    #             flag = marker[i]
    #             i += 200
    #     else:
    #         i += 1   


    # 去除训练数据集中无用的数据
    i = 0
    flag = -1
    while i < len(marker):
        if marker[i] == 0 | marker[i] not in [1,2,3,4,5]:
            i += 1
            flag = -1
        else:
            if marker[i] != flag:
                newData.append(data[i:i+200,:])
                newMarker.append(marker[i:i+200])
                flag = marker[i]
                i += 200
            else:
                i += 1

    # newData = np.array(newData)
    # newMarker = np.ndarray(newMarker)

    print("newData length:", len(newData))
    print("newMarker length:", len(newMarker))

    # 创建一个包含data和marker的字典
    Tr = {'data': newData, 'marker': newMarker}
    scipy.io.savemat(f'./data/training/new_train_data_{n+1}.mat', {'Tr': Tr})
