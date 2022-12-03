import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from mpl_toolkits.mplot3d import Axes3D


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
    data = data[0:173 * (11 + 1)]
    print(data.min(axis=0))
    data = data-data.min(axis=0)
    atom = np.array(atom_list).reshape(-1, 1)
    return data, atom

if __name__ == '__main__':
    path = "D:/DeepLearning/PytorchTest/data/Topololy of solute.xlsx"
    df = pd.read_excel(path)
    Bond_order = df.loc[:, "Bond order"].values
    Bond = df.loc[:, "Bond"].values
    dict = {}
    atom_to_position = {}
    temp_neighbors = torch.zeros(173, 173)
    neighbors = torch.zeros(1, 172)
    atomic_numbers = []
    position = torch.zeros(173, 3)
    # dict = {'S1': 0,'S2': 1,'S3': 2,'S4': 3,'S5': 4,'S6': 5,'S7': 6,'S8': 7,'S9': 8,
    #         'F1':9,'F2':10,'F3':11,'F4':12,'F5':13,'F6':14,}
    for i in range(0,173):
        if (i < 9):
            dict['S' + str(i+1)] = i
        elif (i >=9 and i<15):
            dict['F' + str(i+1-9)] = i
        elif (i >=15 and i<45):
            dict['O' + str(i+1-15)] = i
        elif (i >=45 and i<121):
            dict['C' + str(i+1-45)] = i
        elif (i >=121 and i<173):
            dict['H' + str(i+1-121)] = i
    # print(dict)




    for i in range(173):
        if (i < 9):
            atomic_numbers.append(16)
        elif (i >=9 and i<15):
            atomic_numbers.append(9)
        elif (i >=15 and i<45):
            atomic_numbers.append(8)
        elif (i >=45 and i<121):
            atomic_numbers.append(6)
        elif (i >=121 and i<173):
            atomic_numbers.append(1)
    atomic_numbers = torch.tensor(atomic_numbers)
    # print(atomic_numbers)


    for index in range(0, 1):

        path = "D:/DeepLearning/PytorchTest/data/NMF/NMF100.pdb"
        data,atom = getDataALL(path)
        data = torch.from_numpy(data)
        atom_list = atom.tolist()
        for i in range(173):
            temp = str(atom_list[i])[2:][:-2]
            atom_to_position[temp] = data[i]
        for i in range(173):
            if (i < 9):
                position[i] = atom_to_position['S' + str(i + 1)]
            elif (i >= 9 and i < 15):
                position[i] = atom_to_position['F' + str(i + 1 - 9)]
            elif (i >= 15 and i < 45):
                position[i] = atom_to_position['O' + str(i + 1 - 15)]
            elif (i >= 45 and i < 121):
                position[i] = atom_to_position['C' + str(i + 1 - 45)]
            elif (i >= 121 and i < 173):
                position[i] = atom_to_position['H' + str(i + 1 - 121)]
        for i in range(0, 1):
            temp_data = data[i * 173:173 * (i + 1)]
            temp_atom = atom[i * 173:173 * (i + 1)]
            for connect_atom in Bond:
                left_atom = connect_atom.split("-")[0]
                right_atom = connect_atom.split("-")[-1]
                left_index = np.where(temp_atom == left_atom)[0]
                right_index = np.where(temp_atom == right_atom)[0]
                left_data = temp_data[left_index]
                right_data = temp_data[right_index]
                if (right_atom == 'H2'):
                    print("left_data=",left_data)
                left_atom_value = dict[left_atom]
                right_atom_value = dict[right_atom]
                temp_neighbors[left_atom_value][right_atom_value] = 1
                temp_neighbors[right_atom_value][left_atom_value] = 1
    count = 0
    for i in range(173):
        temp_neighbors_list = []
        index = 0
        for j in range(173):
            if(temp_neighbors[i][j] == 1):
                index = index + 1
                temp_neighbors_list.append(j)
        for k in range(173-index-1):
            temp_neighbors_list.append(0)
        temp_neighbors_tensor = torch.tensor(temp_neighbors_list).reshape(1,172)
        if (count == 0):
            neighbors = temp_neighbors_tensor
        elif (count != 0):
            neighbors = torch.cat([neighbors,temp_neighbors_tensor],dim = 0)
        count = count + 1



