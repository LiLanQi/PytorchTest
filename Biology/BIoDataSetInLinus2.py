from torch.optim.lr_scheduler import LambdaLR
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from tensorboardX import SummaryWriter

import argparse
import os
from Net import Net
import torch
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Finetune')
parser.add_argument('--batchsize', type=int, default=4)
parser.add_argument('--result_src', type=str, default='data/DMF-LJ.xvg')
parser.add_argument('--data_src', type=str, default='data/tempDMF/')
parser.add_argument('--lr', type=float, default=1e-4)
# parser.add_argument('--momentum', type=float, default=0.9)
# parser.add_argument('--decay', type=float, default=5e-4)

DEVICE = torch.device('cuda:0')
args = parser.parse_args()
writer = SummaryWriter('BioLogs2')

# 返回一个结果张量
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


class trainset(Dataset):
    def __init__(self, rootdir, result_src, phase):
        self.root = rootdir
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
        self.result_list = result_list
        self.lables = readFileResult(args.result_src)  # 存入所有mdx 文件对应的output值
        # print("self.lables=",self.lables)

    def __getitem__(self, index):
        result = self.result_list[index]
        lable = self.lables[index]
        return result, lable

    def __len__(self):
        return len(self.result_list)


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
    dict = {'H': 0, 'C': 2, 'O': 0, 'N': 4, 'S': 5,
            'F': 6}  # count0= 4566 countN= 3486 countC= 13194 countF= 216 countS= 324 countH= 26274
    data_loader = load_train(args.data_src, args.batchsize, args.result_src, "train")
    net = Net()
    net.apply(weight_init)
    net = net.train()
    net = net.to(DEVICE)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.99))
    criterion = torch.nn.MSELoss()
    running_loss = 0
    total_loss = 0
    for epoch in range(5):
        countH = 0
        tf_flag = 0
        for i_batch, batch_data in tqdm(enumerate(data_loader)):

            inputs, labels = batch_data
            labels = labels.to(DEVICE)

            # print("inputs大小：",inputs.shape) #torch.Size([batch_size, 16020, 4])
            # print("标签类别大小：", labels.shape)#torch.Size([64])
            # print("inputs=",inputs)
            # print("labels=", labels)
            data = torch.zeros((inputs.shape[0], 80, 160, 70))
            data = data.to(DEVICE)
            for i in range(inputs.shape[0]):
                for j in range(inputs.shape[1]):
                    x = int(inputs[i][j][0])
                    y = int(inputs[i][j][1])
                    z = int(inputs[i][j][2] * 2)
                    atom = chr(int(inputs[i][j][3]))  # 将ascii转换成相应的原子字符
                    if ((x < 160) and (y < 70) and (z < 80)):
                        data[i][z][x][y] = max(dict[atom], data[i][z][x][y])  # 将该区域赋特定的原子值
                        # print("x=",x,"y=",y,"z=",z,"atom=",data[i][z][x][y])

            optimizer.zero_grad()
            # print("data=",data)
            # print("type(data)=",type(data))
            # print("type(inputs)=",type(inputs))
            # with torch.set_grad_enabled(True):  # 当requires_grad设置为False时,反向传播时就不会自动求导了，因此大大节约了显存或者说内存。
            if tf_flag == 0:
                tf_data = data
                tf_labels = labels
            else:
                data = tf_data
                labels = tf_labels
            outputs = net(data).squeeze(-1)
            print("outputs=", outputs)
            print("labels=", labels)
            loss = criterion(outputs, labels)

            # preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()
            # lr_scheduler = LambdaLR(optimizer=optimizer, lr_lambda=poly_lr_scheduler)
            # lr_scheduler.step()
            # print statistics
            running_loss += loss.item()
            total_loss += loss.item()
            writer.add_scalar('train loss', loss.item(), i_batch)
            if (i_batch % 10 == 9):
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i_batch + 1, running_loss / 10))
                running_loss = 0.0
        writer.add_scalar('total loss', total_loss/(i_batch+1), epoch)
    print('BIoDataSetInLinus2 Finished Training, lr=',args.lr,"batchsize=",args.batchsize)
    PATH = './bio_net2.pth'
    torch.save(net.state_dict(), PATH)


