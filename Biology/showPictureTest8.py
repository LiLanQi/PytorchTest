import os

import matplotlib.pyplot as plt
import numpy as np

#返回一个list，list里面为每个状态的张量,暂时先训练10个样本
def getDataTensor(path):
    atom_list = []
    file = open(path, "r", encoding='UTF-8')
    file_list = file.readlines()
    result_list = []
    for i in range(5, len(file_list) - 2):
        temp_data = file_list[i].strip().split('1.00  0.00')[0][26:].strip()
        temp_atom = file_list[i][13]
        atom_list.append(temp_atom)
        temp_data = temp_data.split(' ')
        for j in temp_data:
            if (j == ''):
                continue
            result_list.append(float(j))
    data = np.array(result_list).reshape(-1,3)
    return data,atom_list

if __name__ == '__main__':
    # path_list = readFileData("D:/DeepLearning/PytorchTest/data/tempDMF/")
    # data_tensor_list = getDataTensor(path_list)
    for i in range(120, 950):
        path = "D:/DeepLearning/PytorchTest/data/tempDMF/md" + str(i) + ".pdb"
        # print(path)
        # path_list.append("D:/DeepLearning/PytorchTest/data/tempDMF/md1.pdb")
        data,atom_list = getDataTensor(path)
        xO = []
        yO = []
        zO = []
        xS = []
        yS = []
        zS = []
        xF = []
        yF = []
        zF = []
        xH = []
        yH = []
        zH = []
        xC = []
        yC = []
        zC = []
        xN = []
        yN = []
        zN = []
        for j in range(16020):
           for k in range(3):
               if(atom_list[j] == 'O'):
                   if(k == 0):
                       xO.append(data[j][k])
                   if (k == 1):
                       yO.append(data[j][k])
                   if (k == 2):
                       zO.append(data[j][k])
               if (atom_list[j] == 'S'):
                   if (k == 0):
                       xS.append(data[j][k])
                   if (k == 1):
                       yS.append(data[j][k])
                   if (k == 2):
                       zS.append(data[j][k])
               if (atom_list[j] == 'F'):
                   if (k == 0):
                       xF.append(data[j][k])
                   if (k == 1):
                       yF.append(data[j][k])
                   if (k == 2):
                       zF.append(data[j][k])
               if (atom_list[j] == 'H'):
                   if (k == 0):
                       xH.append(data[j][k])
                   if (k == 1):
                       yH.append(data[j][k])
                   if (k == 2):
                       zH.append(data[j][k])
               if (atom_list[j] == 'N'):
                   if (k == 0):
                       xN.append(data[j][k])
                   if (k == 1):
                       yN.append(data[j][k])
                   if (k == 2):
                       zN.append(data[j][k])
               if (atom_list[j] == 'C'):
                   if (k == 0):
                       xC.append(data[j][k])
                   if (k == 1):
                       yC.append(data[j][k])
                   if (k == 2):
                       zC.append(data[j][k])
        # fig = plt.figure()
        # ax = fig.add_subplot()

        fig = plt.figure(i)  # 新图 0


        # ax.scatter(xO, yO, zO, marker='p')
        # ax.scatter(xS, yS, zS, marker='^')
        # ax.scatter(xH, yH, zH, marker='<')
        # ax.scatter(xN, yN, zN, marker='.')
        # ax.scatter(xC, yC, zC, marker='*')
        # ax.scatter(xF, yF, zF, marker='*')
        # print("count0=",count0,"countN=",countN,"countC=",countC,"countF=",countF,"countS=",countS,"countH=",countH)
        # ax.scatter(xO, yO, zO, c='b')
        # ax.scatter(xS, yS, zS, c='y')
        # ax.scatter(xH, yH, zH, c='g')
        # ax.scatter(xN, yN, zN, c='k')
        # ax.scatter(xC, yC, zC, c='r')
        # ax.scatter(xF, yF, zF, c='w')

        # ax.scatter(xO, yO, c='b')
        # ax.scatter(xS, yS, c='y')
        # ax.scatter(xH, yH, c='g')
        # ax.scatter(xN, yN, c='k')
        # ax.scatter(xC, yC, c='r')
        # ax.scatter(xF, yF, marker='*')

        # pathH = "pictureH/"
        # pathO = "pictureO/"
        # pathS = "pictureS/"
        # pathC = "pictureC/"
        # pathN = "pictureN/"
        # pathF = "pictureF/"

        pathFNS = "D:/DeepLearning/PytorchTest/data/picture/pictureALL2/"

        PointSite = 1.5

        plt.scatter(xF, yF, s=PointSite, c="r")
        plt.scatter(xN, yN, s=PointSite, c="g")
        plt.scatter(xS, yS, s=PointSite, c="y")
        plt.scatter(xC, yC, s=PointSite, c="b")
        plt.scatter(xH, yH, s=PointSite, c="k")
        plt.scatter(xO, yO, s=PointSite, marker='*')
        # 设置标题
        plt.title("ALL atom", fontsize=12)

        plt.xlim((0, 160))
        plt.ylim((0, 70))
        # 设置x轴
        plt.xlabel('X Label', fontsize=12)
        # 设置y轴
        plt.ylabel('Y Label', fontsize=12)

        plt.savefig(pathFNS+ str(i) +'.jpg')
        plt.close(i)  # 关闭图 0

        # ax.set_zlabel('Z Label')
        # plt.show()

