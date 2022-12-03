import argparse
import os

import torch
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='Finetune')
parser.add_argument('--batchsize', type=int, default=8)
parser.add_argument('--result_src', type=str, default='D:/DeepLearning/PytorchTest/data/DMF-LJ.xvg')
parser.add_argument('--data_src', type=str, default='D:/DeepLearning/PytorchTest/data/tempDMF/')
parser.add_argument('--lr', type=float, default=1e-4)

args = parser.parse_args()


#返回一个结果张量
def readFileResult(root_dir):
    file = open(root_dir, "r", encoding='UTF-8')
    file_list = file.readlines()
    result_list = []
    for i in range(19, len(file_list)):
        temp_data = file_list[i].strip().split('  ')[-1]
        temp_data = float(temp_data)
        result_list.append(temp_data)
    result = torch.tensor(result_list)
    return result

def readFileData(root_dir):
    path_list = []
    for i in range(0,100000):
        temp_path = "md" + str(i) + ".pdb"
        path = os.path.join(root_dir,temp_path)
        path_list.append(path)
   #path_list.append(os.path.join(root_dir,"md/origin.pdb")) 暂时不需要
    return path_list

#返回一个list，list里面为每个状态的张量,暂时先训练10个样本
def getDataTensor(path_list):
    data_tensor_list = []
    index = 0
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
        index = index + 1
        if (index == 10):
            break
    return data_tensor_list
if __name__ == '__main__':
    path_list = readFileData(args.data_src)
    data_tensor_list = getDataTensor(path_list)
    print(data_tensor_list[0])
    print(data_tensor_list[1])
    print(len(data_tensor_list))



