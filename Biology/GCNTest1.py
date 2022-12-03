#encoding=utf-8
import pandas as pd
from torch.optim.lr_scheduler import LambdaLR
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from tensorboardX import SummaryWriter
# import pandas as pd
import argparse
import os
# from MyNet3D import MyNet3D
import torch
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Finetune')
parser.add_argument('--batchsize', type=int, default=4)
parser.add_argument('--result_src', type=str, default='../data/newACE/ACE-LJ.xvg')
parser.add_argument('--data_src', type=str, default='../data/newACE/ACE_pdb')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--topological_solute_path', type=str, default='../data/Topololy of solute.xlsx')
# parser.add_argument('--momentum', type=float, default=0.9)
# parser.add_argument('--decay', type=float, default=5e-4)

DEVICE = torch.device('cuda:0')
args = parser.parse_args()
writer = SummaryWriter('GCNTest1')

dict_solute = {}
dict_solvent = {}

def get_solute_position(rootdir):
    path_list = readFileData(rootdir)
    for path in path_list:
        file = open(path, "r")
        file_list = file.readlines()
        for i in range(5, 173 + 5):
            temp_data = file_list[i].strip().split('1.00  0.00')[0][26:].strip()
            temp_data = temp_data.split('.')
            temp_atom = file_list[i][13:16].strip()  # 该点所对应的原子
            x = float(temp_data[0] + "." + temp_data[1][0:3])
            y = float(temp_data[1][3:].strip() + "." + temp_data[2][0:3])
            z = float(temp_data[2][3:].strip() + "." + temp_data[3][0:3])
            dict_solute[temp_atom] = i-5
        break
    return dict_solute


# 返回一个结果张量
def readFileResult(root_dir):
    file = open(root_dir, "r")
    file_list = file.readlines()
    result_list = []
    for i in range(0, 100):
        temp_data = file_list[i].strip().split('  ')[-1].strip()
        temp_data = float(temp_data)
        result_list.append(temp_data)
    result = torch.tensor(result_list)
    return result


def readFileData(root_dir):
    path_list = []
    for i in range(0, 100):
        temp_path = "ACE" + str(i) + ".pdb"
        path = os.path.join(root_dir, temp_path)
        path_list.append(path)
    # path_list.append(os.path.join(root_dir,"md/origin.pdb")) 暂时不需要
    return path_list

def get_solute_adj(path):
    solute_adj = torch.zero(173,173)
    df = pd.read_excel(path)
    Bond_order = df.loc[:, "Bond order"].values
    Bond = df.loc[:, "Bond"].values
    for connect_atom in Bond:
        left_atom = connect_atom.split("-")[0]
        right_atom = connect_atom.split("-")[-1]
        left_atom_value = dict_solute[left_atom]
        right_atom_value = dict_solute[right_atom]
        print("left_atom_value=",left_atom_value,"right_atom_value=",right_atom_value)
        solute_adj[left_atom_value][right_atom_value] = 1
        solute_adj[right_atom_value][left_atom_value] = 1
    return solute_adj


class trainset(Dataset):
    def __init__(self, rootdir, result_src, phase):
        self.root = rootdir
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
                temp_atom = file_list[i][13:16].strip()  # 该点所对应的原子
                x = float(temp_data[0] + "." + temp_data[1][0:3])
                y = float(temp_data[1][3:].strip() + "." + temp_data[2][0:3])
                z = float(temp_data[2][3:].strip() + "." + temp_data[3][0:3])
                # min_x = min(x, min_x)
                # min_y = min(y, min_y)
                # min_z = min(z, min_z)
                #
                temp_result_list.append(x)
                temp_result_list.append(y)
                temp_result_list.append(z)
                # temp_result_list.append(ord(temp_atom))  # 存入该字符对应的ascii码
                # temp_result_list.append(temp_atom)
            temp = torch.tensor(temp_result_list).reshape(-1, 3)
            solute_data = temp[:173*12, :]
            solvent_data = temp[173*12:, :]
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
    dict_solute = get_solute_position(args.data_src)
    solute_adj = get_solute_adj(args.topological_solute_path)
    print("solute_adj=",solute_adj)
    # data_loader = load_train(args.data_src, args.batchsize, args.result_src, "train")
    # net = MyNet3D()
    # net.eval()
    # net = net.to(DEVICE)


    # optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.99))
    # criterion = torch.nn.MSELoss()
    # loss_list = []
    # for epoch in range(100):
    #     # tf_flag = 0
    #     total_loss = 0
    #     running_loss = 0
    #     count = 0
    #     for i_batch, batch_data in tqdm(enumerate(data_loader)):
    #         count = count + 1
    #         solute_data, solvent_data, labels = batch_data
    #         labels = labels.to(DEVICE)
    #         solute_data = solute_data.to(DEVICE)
    #         solvent_data = solvent_data.to(DEVICE)
    #         data = torch.cat((solute_data, solvent_data), 4)
    #         optimizer.zero_grad()
    #         with torch.set_grad_enabled(True):  # 当requires_grad设置为False时,反向传播时就不会自动求导了，因此大大节约了显存或者说内存。
    #             outputs = net(data).squeeze(-1)
    #             loss = criterion(outputs, labels)
    #         print("outputs=", outputs)
    #         print("labels=", labels)
    #         print("loss=", loss)
    #         # preds = torch.max(outputs, 1)
    #         loss.backward()
    #         optimizer.step()
    #         # lr_scheduler = LambdaLR(optimizer=optimizer, lr_lambda=poly_lr_scheduler)
    #         # lr_scheduler.step()
    #         # print statistics
    #         running_loss += loss.item()
    #         total_loss += loss.item() * data.size(0)
    #         writer.add_scalar('train loss', loss.item(), i_batch)
    #         if (i_batch % 10 == 9):
    #             print('[%d, %5d] loss: %.3f' %
    #                   (epoch + 1, i_batch + 1, running_loss / 10))
    #             running_loss = 0.0
    #     writer.add_scalar('total loss', total_loss / len(data_loader), epoch)
    #     loss_list.append(total_loss / len(data_loader))
    # print("loss_list = ", loss_list)
    # print('GCNTest1 Finished Training, lr=', args.lr, "batchsize=", args.batchsize)
    # PATH = './GCNTest1.pth'
    # torch.save(net.state_dict(), PATH)


