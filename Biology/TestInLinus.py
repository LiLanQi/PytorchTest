import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
from torch.autograd import Variable
from Net import Net
# data1 = torch.rand((64,16020, 3))
# data2 = torch.rand((64,1,200,200,200))
# print(data1.shape)
#
# for i in range(64):
#     for j in range(16020):
#         x = int(data1[i][j][0])
#         y = int(data1[i][j][1])
#         z = int(data1[i][j][2])
#         data2[i][0][x][y][z] += 1

# print(data2)
# net = Net()
# output = net(data2)


import matplotlib.pyplot as plt


def readFileData(root_dir):
    path_list = []
    for i in range(0,1000):
        temp_path = "md" + str(i) + ".pdb"
        path = os.path.join(root_dir,temp_path)
        path_list.append(path)
   #path_list.append(os.path.join(root_dir,"md/origin.pdb")) 暂时不需要
    return path_list


def getDataTensor(path_list):
    data_tensor_list = []
    for path in path_list:
        file = open(path, "r", encoding='UTF-8')
        file_list = file.readlines()
        result_list = []
        for i in range(5, len(file_list) - 2):
            temp_data = file_list[i].strip().split('1.00  0.00')[0][26:].strip()
            temp_data = temp_data.split(' ')
            for j in temp_data:
                if (j == ''):
                    continue
                result_list.append(float(j))
        data_tensor_list.append(torch.tensor(result_list).reshape(-1, 3))

    return data_tensor_list

if __name__ == '__main__':
    path_list = readFileData("data/tempDMF/")
    data_tensor_list = getDataTensor(path_list)
    x = []
    y = []
    z = []
    for i in range(1000):
       data = data_tensor_list[i]
       for j in range(10620):
           for k in range(3):
               if(k == 0):
                   x.append(data[j][k])
               if (k == 1):
                   x.append(data[j][k])
               if (k == 2):
                   x.append(data[j][k])



    fig = plt.figure()
    fig.add_subplot(111, projection='3d')
    plt.scatter(x, y, z)
    plt.show()

