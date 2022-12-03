
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import argparse
import os
from Net import Net
import torch
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Finetune')
parser.add_argument('--batchsize', type=int, default=64)
parser.add_argument('--result_src', type=str, default='D:/DeepLearning/PytorchTest/data/DMF-LJ.xvg')
parser.add_argument('--data_src', type=str, default='D:/DeepLearning/PytorchTest/data/tempDMF/')
parser.add_argument('--lr', type=float, default=1e-4)


args = parser.parse_args()


#返回一个结果张量
def readFileResult(root_dir):
    print(111)
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
    for i in range(0,1000):
        temp_path = "md" + str(i) + ".pdb"
        path = os.path.join(root_dir,temp_path)
        path_list.append(path)
   #path_list.append(os.path.join(root_dir,"md/origin.pdb")) 暂时不需要
    return path_list

class trainset(Dataset):
    def __init__(self, rootdir, target_dir, phase):
        self.root = rootdir
        self.target_dir = target_dir
        # index = 0
        result_list = []
        path_list = readFileData(rootdir) #得到所有mdx.pdb文件的list
        for path in path_list:
            file = open(path, "r", encoding='UTF-8')
            file_list = file.readlines()
            temp_result_list = []
            for i in range(5, len(file_list) - 2):
                temp_data = file_list[i].strip().split('1.00  0.00')[0][26:].strip()
                temp_data = temp_data.split(' ')
                temp_atom = file_list[i][13] #该点所对应的原子
                for j in temp_data:
                    if (j == ''):
                        continue
                    temp_result_list.append(float(j))
                temp_result_list.append(ord(temp_atom)) #存入该字符对应的ascii码
            temp = torch.tensor(temp_result_list).reshape(-1,4)
            result_list.append(temp)
            # index = index + 1
            # if (index == 3):
            #     break
        self.result_list = result_list
        self.lables = readFileResult(self.target_dir)

    def __getitem__(self, index):
        result = self.result_list[index]
        lables = self.lables
        lable = lables[index]
        return result,lable

    def __len__(self):
        return len(self.result_list)


def load_train(root_path, batch_size, target_dir,phase):
    data = trainset(root_path, target_dir, phase)
    print("data=", data)

    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=12)
    return loader




if __name__ == '__main__':
    dict = {'H': 1, 'C': 2, 'O': 3, 'N': 4, 'S': 5, 'F': 6}
    data_loader = load_train(args.data_src, args.batchsize, args.result_src,"train")
    net = Net()
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()
    correct = 0
    for epoch in range(2):
        for i_batch, batch_data in tqdm(enumerate(data_loader)):
            inputs, labels = batch_data
            print("inputs大小：",inputs.shape) #torch.Size([64, 16020, 4])
            print("标签类别大小：", labels.shape)#torch.Size([64])

            data = torch.zeros((inputs.shape[0],80, 160, 70))
            for i in range(inputs.shape[0]):
                for j in range(inputs.shape[1]):
                    x = int(inputs[i][j][0])
                    y = int(inputs[i][j][1])
                    z = int(inputs[i][j][2]*2)
                    atom = chr(int(inputs[i][j][3])) #将ascii转换成相应的原子字符
                    if((x<160) and (y<70) and (z<80)):
                        data[i][z][x][y] = max(dict[atom],data[i][z][x][y]) #讲该区域赋特定的原子值

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):  # 当requires_grad设置为False时,反向传播时就不会自动求导了，因此大大节约了显存或者说内存。
                outputs = net(data)
                loss = criterion(outputs, labels)
            preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()
            correct += torch.sum(preds == labels.data)
            # print statistics
            running_loss += loss.item()
            if i_batch % 10 == 9:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i_batch + 1, running_loss / 10))
                running_loss = 0.0
    print('Finished Training')


