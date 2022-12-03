import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
#DMF1267中的所有溶质、溶剂图化作单独的俩个图
def getDataALL(path):
    atom_list = []
    file = open(path, "r", encoding='UTF-8')
    file_list = file.readlines()
    result_list = []
    for i in range(5, len(file_list) - 2):
        temp_data = file_list[i].strip().split('1.00  0.00')[0][26:].strip()
        temp_atom = file_list[i][13:16].strip()
        atom_list.append(temp_atom)
        temp_data = temp_data.split(' ')
        for j in temp_data:
            if (j == ''):
                continue
            result_list.append(float(j))
    data = np.array(result_list).reshape(-1, 3)
    print("data=", data)
    print("data.shape=", data.shape)
    solute_data = data[0:173 * (11 + 1)]
    solvent_data = data[173 * (11 + 1):]
    # print(data.min(axis=0))
    # data = data-data.min(axis=0)
    atom = np.array(atom_list).reshape(-1, 1)
    solute_atom = atom[:173]
    solvent_atom = atom[173 * 12:173 * 12+12]
    return solute_data, solute_atom, solvent_data, solvent_atom

if __name__ == '__main__':
    path = "D:/DeepLearning/PytorchTest/data/Topololy of solute.xlsx"
    df = pd.read_excel(path)
    Bond_order = df.loc[:, "Bond order"].values
    Bond = df.loc[:, "Bond"].values
    dict = {'H': 1, 'C': 6, 'O': 8, 'N': 7, 'S': 16,
            'F': 9}

    for index in range(0, 1):

        path = "C:/Users/Administrator/Desktop/分子数据/DMF1267.pdb"
        fig = plt.figure(index)  # 新图 0
        ax = fig.add_subplot(projection='3d')
        PointSite = 3.5

        solute_data, solute_atom, solvent_data, solvent_atom = getDataALL(path)

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        for i in range(0, 12):
            temp_data = solute_data[i * 173:173 * (i + 1)]
            temp_atom = solute_atom[i * 173:173 * (i + 1)]
            ax.scatter(temp_data[:, 0], temp_data[:, 1], temp_data[:, 2], s=PointSite, c="r")
            for connect_atom in Bond:
                left_atom = connect_atom.split("-")[0]
                right_atom = connect_atom.split("-")[-1]
                left_index = np.where(temp_atom == left_atom)[0]
                right_index = np.where(temp_atom == right_atom)[0]
                left_data = temp_data[left_index]
                right_data = temp_data[right_index]
                # if (right_atom == 'H2'):
                #     print("left_data=",left_data)
                temp_temp_data = np.vstack((left_data, right_data))
                if (i == 0):
                    color = "cyan"
                if (i == 1):
                    color = "y"
                if (i == 2):
                    color = "b"
                if (i == 3):
                    color = "g"
                if (i == 4):
                    color = "k"
                if (i == 5):
                    color = "pink"
                if (i == 6):
                    color = "violet"
                if (i == 7):
                    color = "gray"
                if (i == 8):
                    color = "brown"
                if (i == 9):
                    color = "chocolate"
                if (i == 10):
                    color = "gold"
                if (i == 11):
                    color = "orange"
                if (i == 12):
                    color = "purple"
                ax.plot(temp_temp_data[:, 0], temp_temp_data[:, 1], temp_temp_data[:, 2], c=color)
        plt.show()
        plt.close(index)