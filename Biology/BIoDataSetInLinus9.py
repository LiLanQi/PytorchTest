from torch.optim.lr_scheduler import LambdaLR
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from tensorboardX import SummaryWriter

import argparse
import os
from MyNet9 import MyNet9
import torch
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Finetune')
parser.add_argument('--batchsize', type=int, default=64)
parser.add_argument('--result_src', type=str, default='../data/newACE/ACE-LJ.xvg')
parser.add_argument('--data_src', type=str, default='../data/newACE/ACE_pdb')
parser.add_argument('--lr', type=float, default=1e-3)
# parser.add_argument('--momentum', type=float, default=0.9)
# parser.add_argument('--decay', type=float, default=5e-4)

DEVICE = torch.device('cuda:1')
args = parser.parse_args()
writer = SummaryWriter('BioLogs9')

# 返回一个结果张量
def readFileResult(root_dir):
    file = open(root_dir, "r", encoding='UTF-8')
    file_list = file.readlines()
    result_list = []
    for i in range(0, 200):
        temp_data = file_list[i].strip().split('  ')[-1].strip()
        temp_data = float(temp_data)
        result_list.append(temp_data)
    result = torch.tensor(result_list)
    return result


def readFileData(root_dir):
    path_list = []
    for i in range(0, 200):
        temp_path = "ACE" + str(i) + ".pdb"
        path = os.path.join(root_dir, temp_path)
        path_list.append(path)
    # path_list.append(os.path.join(root_dir,"md/origin.pdb")) 暂时不需要
    return path_list


class trainset(Dataset):
    def __init__(self, rootdir, result_src, phase):
        self.root = rootdir
        dict = {'H': 0, 'C': 1, 'O': 0, 'N': 2, 'S': 3,
                'F': 4}  # count0= 4566 countN= 3486 countC= 13194 countF= 216 countS= 324 countH= 26274
        solute_list = []
        solvent_list = []
        path_list = readFileData(rootdir)  # 得到所有mdx.pdb文件的list
        for path in path_list:
            file = open(path, "r", encoding='UTF-8')
            file_list = file.readlines()
            temp_result_list = []
            min_x = 0.0
            min_y = 0.0
            min_z = 0.0
            for i in range(5, len(file_list) - 2):
                temp_data = file_list[i].strip().split('1.00  0.00')[0][26:].strip()
                temp_data = temp_data.split('.')
                temp_atom = file_list[i][13]  # 该点所对应的原子
                x = float(temp_data[0] + "." + temp_data[1][0:3])
                y = float(temp_data[1][3:].strip() + "." + temp_data[2][0:3])
                z = float(temp_data[2][3:].strip() + "." + temp_data[3][0:3])
                min_x = min(x, min_x)
                min_y = min(y, min_y)
                min_z = min(z, min_z)
                temp_result_list.append(x)
                temp_result_list.append(y)
                temp_result_list.append(z)
                temp_result_list.append(ord(temp_atom))  # 存入该字符对应的ascii码
            temp = torch.tensor(temp_result_list).reshape(-1, 4)
            solute_data = torch.zeros((170, 140, 80))
            solvent_data = torch.zeros((170, 140, 80))
            for i in range(temp.shape[0]):
                x = int(temp[i][0]) - int(min_x)
                y = int(temp[i][1]) - int(min_y)
                z = int(temp[i][2] * 2) - int(min_z)*2
                atom = chr(int(temp[i][3]))
                if ((x < 170) and (y < 80) and (z<140) and (i<2081)):
                    solute_data[x][z][y] = max(dict[atom], solute_data[x][z][y])
                if ((x < 170) and (y < 80) and (z<140) and (i>=2081)):
                    solvent_data[x][z][y] = max(dict[atom], solvent_data[x][z][y])
            solute_list.append(solute_data)  # 存入每一个mdx 文件的所有原子坐标，直到所有mdx 文件的所有原子坐标都存入进来
            solvent_list.append(solvent_data)
        self.solute_list = solute_list
        self.solvent_list = solvent_list
        self.lables = readFileResult(args.result_src)  # 存入所有mdx 文件对应的output值
        # print("self.lables=",self.lables)

    def __getitem__(self, index):
        solute_list = self.solute_list[index]
        solvent_list = self.solvent_list[index]
        lable = self.lables[index]
        return solute_list,solvent_list,lable

    def __len__(self):
        return len(self.solute_list)


def load_train(root_path, batch_size, result_src, phase):
    data = trainset(root_path, result_src, phase)
    print("data=", data)

    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=12)
    return loader


def poly_lr_scheduler(epoch, num_epochs=300, power=0.9):
    return (1 - epoch / num_epochs) ** power


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    data_loader = load_train(args.data_src, args.batchsize, args.result_src, "train")
    net = MyNet9()
    net.apply(weight_init)
    net = net.train()
    net = net.to(DEVICE)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.99))
    criterion = torch.nn.MSELoss()
    loss_list = []
    for epoch in range(100):
        # tf_flag = 0
        total_loss = 0
        running_loss = 0
        count = 0
        for i_batch, batch_data in tqdm(enumerate(data_loader)):
            count = count + 1
            solute_data,solvent_data, labels = batch_data
            labels = labels.to(DEVICE)
            solute_data = solute_data.to(DEVICE)
            solvent_data = solvent_data.to(DEVICE)
            # print("solute_data.shape",solute_data.shape)#(B,160,80,70)
            # print("solvent_data.shape", solvent_data.shape)#(B,160,80,70)
            data = torch.cat((solute_data,solvent_data),3)
            # print("data.shape", data.shape)  # (B,160,80,140)
            # print("标签类别大小：", labels.shape)#torch.Size([B])
            # print("inputs=",inputs)
            # print("labels=", labels)

            optimizer.zero_grad()
            with torch.set_grad_enabled(True):  # 当requires_grad设置为False时,反向传播时就不会自动求导了，因此大大节约了显存或者说内存。
                outputs = net(data).squeeze(-1)
                loss = criterion(outputs, labels)
            print("outputs=",outputs)
            print("labels=", labels)
            print("loss=",loss)
            # preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()
            # lr_scheduler = LambdaLR(optimizer=optimizer, lr_lambda=poly_lr_scheduler)
            # lr_scheduler.step()
            # print statistics
            total_loss += loss.item()*data.size(0)
        writer.add_scalar('total loss', total_loss/len(data_loader), epoch)
        loss_list.append(total_loss/len(data_loader))
    print("loss_list = ",loss_list)
    print('BIoDataSetInLinus9 Finished Training, lr=',args.lr,"batchsize=",args.batchsize)
    PATH = './bio_net9.pth'
    torch.save(net.state_dict(), PATH)


