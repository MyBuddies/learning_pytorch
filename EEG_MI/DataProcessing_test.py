import scipy.io

'''
# 测试数据集数据预处理
'''
for n in range(6):
    # 读取.mat文件
    mat_data = scipy.io.loadmat(f'./data/testing/test_subject_{n+1}.mat')

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


    # print(marker[71257])
    # print(data[71257])
    newData = []
    newMarker = [] 
    
    # for i in range(len(marker)):
    #     if marker[i] in [1,2,3,4,5]:
    #         newData.append(data[i])
    #         newMarker.append(marker[i])

    # 去除测试数据集中无用的数据
    i = 0
    flag = -1
    while i < len(marker):
        if marker[i] == 0:
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
    Te = {'data': newData, 'marker': newMarker}
    scipy.io.savemat(f'./data/testing/new_test_data_{n+1}.mat', {'Te': Te})
