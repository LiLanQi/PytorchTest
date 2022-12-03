import argparse
import os
import torch

parser = argparse.ArgumentParser(description='Finetune')
parser.add_argument('--batchsize', type=int, default=64)
parser.add_argument('--result_src', type=str, default='D:/DeepLearning/PytorchTest/data/DMF-LJ.xvg')
parser.add_argument('--data_src', type=str, default='data/tempDMF/')
parser.add_argument('--lr', type=float, default=1e-4)
args = parser.parse_args()

def readFileResult(root_dir):
    file = open(root_dir, "r", encoding='UTF-8')
    file_list = file.readlines()
    result_list = []
    for i in range(19, 1019):
        temp_data = file_list[i].strip().split('  ')[-1].strip()
        temp_data = float(temp_data)
        result_list.append(temp_data)
    result = torch.tensor(result_list)
    return result


def readFileData(root_dir):
    path_list = []
    for i in range(0, 1000):
        temp_path = "md" + str(i) + ".pdb"
        path = os.path.join(root_dir, temp_path)
        path_list.append(path)
    # path_list.append(os.path.join(root_dir,"md/origin.pdb")) 暂时不需要
    return path_list



def readFile(rootdir):
    result_list = []
    path_list = readFileData(rootdir)  # 得到所有mdx.pdb文件的list
    for path in path_list:
        file = open(path, "r", encoding='UTF-8')
        file_list = file.readlines()
        temp_result_list = []
        for i in range(5, len(file_list) - 2):
            temp_data = file_list[i].strip().split('1.00  0.00')[0][26:].strip()
            temp_data = temp_data.split(' ')
            temp_atom = file_list[i][13]  # 该点所对应的原子
            for j in temp_data:
                if (j == ''):
                    continue
                temp_result_list.append(float(j))
            temp_result_list.append(ord(temp_atom))  # 存入该字符对应的ascii码
        temp = torch.tensor(temp_result_list).reshape(-1, 4)
        result_list.append(temp)  # 存入每一个mdx 文件的所有原子坐标，直到所有mdx 文件的所有原子坐标都存入进来
    return result_list

if __name__ == '__main__':
    inputs = readFile(args.data_src) #1000
    for i in range(len(inputs)):


