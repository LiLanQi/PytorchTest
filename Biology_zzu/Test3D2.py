#encoding=utf-8
from torch.optim.lr_scheduler import LambdaLR
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from tensorboardX import SummaryWriter

import argparse
import os
from MyNet3D import MyNet3D
import torch
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Finetune')
parser.add_argument('--batchsize', type=int, default=4)
parser.add_argument('--result_src', type=str, default='../data/newACE/ACE-LJ.xvg')
parser.add_argument('--data_src', type=str, default='../data/newACE/ACE_pdb')
parser.add_argument('--lr', type=float, default=1e-3)
# parser.add_argument('--momentum', type=float, default=0.9)
# parser.add_argument('--decay', type=float, default=5e-4)

DEVICE = torch.device('cuda:0')
args = parser.parse_args()
writer = SummaryWriter('BioLogs3D2')


# 返回一个结果张量
def readFileResult(root_dir):
    file = open(root_dir, "r")
    file_list = file.readlines()
    result_list = []
    for i in range(200, 600):
        temp_data = file_list[i].strip().split('  ')[-1].strip()
        temp_data = float(temp_data)
        result_list.append(temp_data)
    result = torch.tensor(result_list)
    return result


def readFileData(root_dir):
    path_list = []
    for i in range(200, 600):
        temp_path = "ACE" + str(i) + ".pdb"
        path = os.path.join(root_dir, temp_path)
        path_list.append(path)
    # path_list.append(os.path.join(root_dir,"md/origin.pdb")) 暂时不需要
    return path_list


class trainset(Dataset):
    def __init__(self, rootdir, result_src, phase):
        self.root = rootdir
        dict = {'H': 0, 'C': 1, 'O': 2, 'N': 3, 'S': 4,
                'F': 5}  # count0= 4566 countN= 3486 countC= 13194 countF= 216 countS= 324 countH= 26274
        solute_list = []
        solvent_list = []
        path_list = readFileData(rootdir)  # 得到所有mdx.pdb文件的list
        for path in path_list:
            file = open(path, "r")
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
            solute_data = torch.zeros((6, 170, 80, 140))
            solvent_data = torch.zeros((6, 170, 80, 140))
            for i in range(temp.shape[0]):
                x = int(temp[i][0]) - int(min_x)
                y = int(temp[i][1]) - int(min_y)
                z = int(temp[i][2] * 2) - int(min_z) * 2
                pos_atom = dict[chr(int(temp[i][3]))]
                # print("atom_data=",atom_data)
                # print("type(atom_data)=", type(atom_data))
                # # print("solute_data[pos_atom][x][y][z]=",solute_data[pos_atom][x][y][z])
                # print("x=",x,"y=",y,"z=",z)
                # print("type(x)=", type(x), "type(y)=", type(y), "type(z)=", type(z))
                if ((x < 170) and (y < 80) and (z < 140) and (i < 2081)):
                    solute_data[pos_atom][x][y][z] = max(pos_atom, solute_data[pos_atom][x][y][z])
                if ((x < 170) and (y < 80) and (z < 140) and (i >= 2081)):
                    solvent_data[pos_atom][x][y][z] = max(pos_atom, solvent_data[pos_atom][x][y][z])
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
        return solute_list, solvent_list, lable

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
    load_PATH = './bio_net3d.pth'
    data_loader = load_train(args.data_src, args.batchsize, args.result_src, "train")
    net = MyNet3D()
    net.load_state_dict(torch.load(load_PATH))
    net.eval()
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
            solute_data, solvent_data, labels = batch_data
            labels = labels.to(DEVICE)
            solute_data = solute_data.to(DEVICE)
            solvent_data = solvent_data.to(DEVICE)
            data = torch.cat((solute_data, solvent_data), 4)
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):  # 当requires_grad设置为False时,反向传播时就不会自动求导了，因此大大节约了显存或者说内存。
                outputs = net(data).squeeze(-1)
                loss = criterion(outputs, labels)
            print("outputs=", outputs)
            print("labels=", labels)
            print("loss=", loss)
            # preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()
            # lr_scheduler = LambdaLR(optimizer=optimizer, lr_lambda=poly_lr_scheduler)
            # lr_scheduler.step()
            # print statistics
            running_loss += loss.item()
            total_loss += loss.item() * data.size(0)
            writer.add_scalar('train loss', loss.item(), i_batch)
            if (i_batch % 10 == 9):
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i_batch + 1, running_loss / 10))
                running_loss = 0.0
        writer.add_scalar('total loss', total_loss / len(data_loader), epoch)
        loss_list.append(total_loss / len(data_loader))
    print("loss_list = ", loss_list)
    print('Test3D2 Finished Training, lr=', args.lr, "batchsize=", args.batchsize)
    PATH = './bio_net3d2.pth'
    torch.save(net.state_dict(), PATH)


