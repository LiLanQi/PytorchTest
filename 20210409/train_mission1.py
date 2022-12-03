from tqdm import tqdm
import argparse
#import data_loader
from Dogset import load_train
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import time
from torchsummary import summary
import datetime
from tensorboardX import SummaryWriter
import os
#命令行参数设定
parser = argparse.ArgumentParser(description='Finetune')
parser.add_argument('--model', type=str, default='resnext')
parser.add_argument('--batchsize', type=int, default=64)
parser.add_argument('--src', type=str, default='data/low-resolution')
parser.add_argument('--tar', type=str, default='TEST_A')
parser.add_argument('--n_class', type=int, default=131)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--n_epoch', type=int, default=30)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--decay', type=float, default=5e-4)
parser.add_argument('--data', type=str, default='')
parser.add_argument('--early_stop', type=int, default=20)
parser.add_argument('--trainlst', type=str, default='TrainAndValList/train.lst')
parser.add_argument('--vallst', type=str, default='TrainAndValList/validation.lst')
args = parser.parse_args()

# Parameter setting
DEVICE = torch.device('cuda:0')
BATCH_SIZE = {'src': int(args.batchsize), 'tar': int(args.batchsize)}
writer = SummaryWriter('logs')

def load_model():
    model = torchvision.models.vgg19(pretrained=True)
    n_features = model.classifier[6].in_features
    fc = torch.nn.Linear(n_features, args.n_class)
    model.classifier[6] = fc
    model.classifier[6].weight.data.normal_(0,0.005)
    model.classifier[6].bias.data.fill_(0.1)
    return model

def get_optimizer(model):
    learning_rate = args.lr
    '''param_group = [
        {'params': model.features.parameters(), 'lr': learning_rate}]
    for i in range(6):
        param_group += [{'params': model.classifier[i].parameters(),
                         'lr': learning_rate}]
    param_group += [{'params': model.classifier[6].parameters(),
                     'lr': learning_rate}]
    optimizer = optim.SGD(param_group, momentum=args.momentum)'''
    optimizer = optim.SGD(model.parameters(),lr=args.lr, momentum=args.momentum)
    return optimizer

#对于怎么调整学习率，还得再看下.
def lr_schedule(optimizer, epoch):
    if epoch % 10 != 0:
        return
    for i in range(len(optimizer.param_groups)):
        optimizer.param_groups[i]['lr'] =  optimizer.param_groups[i]['lr'] * 0.7 #是不是每一次都有一个单独的lr

def train(model, dataloaders, optimizer, criterion, epoch):
    total_loss, correct = 0, 0
    for inputs, labels in tqdm(dataloaders['train']):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        with torch.set_grad_enabled(True): #当requires_grad设置为False时,反向传播时就不会自动求导了，因此大大节约了显存或者说内存。
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        preds = torch.max(outputs, 1)[1] #torch.max()[0]， 只返回最大值的每个数   troch.max()[1]， 只返回最大值的每个索引
        print("output为",outputs)
        print("prds为",preds)
        print("labels为",labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0) #2.这里loss应该有很多行数据？？？？
        correct += torch.sum(preds == labels.data)
    train_loss = total_loss/len(dataloaders['train']) #3.为什么，难道算的是一个batch的loss，应该是
    train_acc = correct.double() / len(dataloaders['train'].dataset)
    print(datetime.datetime.today(), 'Epoch: [{:02d}/{:02d}]---train, train_Loss: {:.6f}, train_Acc: {:.4f}'.format(epoch, args.n_epoch, train_loss, train_acc))
    writer.add_scalar('train loss', train_loss, epoch) #第一个参数是名称，第二个参数是Y轴，第三个参数是X轴，类型是scalar，将其保存在文件里
    writer.add_scalar('train acc', train_acc, epoch)
    os.makedirs(os.path.join("./", args.model), exist_ok=True)
    torch.save(model.state_dict(), os.path.join("./mission1_models/", str(epoch)+'model.pkl'))
    return train_acc

def val(model, dataloaders, optimizer, criterion, best_acc, epoch):
    total_loss, correct = 0, 0
    for inputs, labels in tqdm(dataloaders['val']):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        preds = torch.max(outputs, 1)[1]
        total_loss += loss.item() * inputs.size(0)
        correct += torch.sum(preds == labels.data)
    val_loss = total_loss/len(dataloaders['val'])
    val_acc = correct.double() / len(dataloaders['val'].dataset)
    print(datetime.datetime.today(), 'Epoch: [{:02d}/{:02d}]---val, val_Loss: {:.6f}, val_Acc: {:.4f}'.format(epoch, args.n_epoch, val_loss, val_acc))
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), os.path.join("./" , 'val_model.pkl'))
    return best_acc


def mission(model, dataloaders, optimizer):
    since = time.time()
    best_acc = 0
    criterion = nn.CrossEntropyLoss()

    for epoch in range(30):
        lr_schedule(optimizer, epoch)
        # 训练部分
        model.train() #将模型设置为训练状态
        train_acc = train(model, dataloaders, optimizer, criterion, epoch)
        if train_acc> best_acc:
            best_acc = train_acc

        #val部分
        model.eval()#将模型设置为测试状态，为此种模式的时候，模型内部的权值不会改变
        best_acc = val(model, dataloaders, optimizer, criterion, best_acc, epoch)

    time_pass = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_pass // 60, time_pass % 60))
    return model, best_acc


        #主程序
if __name__ == '__main__':
    torch.manual_seed(10)

    # val dataset
    root_dir = os.path.join(args.data, args.src) #    Dogs/low-resolution
    eval_loader = load_train(root_dir, args.vallst, BATCH_SIZE['src'], 'val') #BATCH_SIZE = 64
    train_loader = load_train(root_dir, args.trainlst, BATCH_SIZE['src'], 'train') #BATCH_SIZE = 64

    # Load model
    model = load_model().to(DEVICE)

    # summary(model, (3, 448, 448))
    print(datetime.datetime.today(),
          ' Train:{}, val: {} '.format(len(train_loader), len(eval_loader))) #1020 82
    optimizer = get_optimizer(model)
    model_best, best_acc = mission(model, {"train": train_loader, "val": eval_loader}, optimizer)
    print("len(train_loader)=", len(train_loader))  # 1020
    print("len(train_loader.dataset)=", len(train_loader.dataset))  # 65228 train.lst中的数量
    print('Best acc: {}'.format(best_acc))
