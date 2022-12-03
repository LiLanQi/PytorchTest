import numpy as np
import torchvision
from torch.utils.data import DataLoader
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from MyDataset import MyDataset
from tqdm import tqdm
from tensorboardX import SummaryWriter

def load_model():
    model = torchvision.models.vgg19(pretrained=True)
    n_features = model.classifier[6].in_features
    fc = torch.nn.Linear(n_features, 131)
    model.classifier[6] = fc
    model.classifier[6].weight.data.normal_(0,0.005)
    model.classifier[6].bias.data.fill_(0.1)
    return model



if __name__=='__main__':
    torch.manual_seed(10)
    DEVICE = torch.device('cuda:0')
    writer = SummaryWriter('mylogs')
    train_augmentation = torchvision.transforms.Compose([torchvision.transforms.Resize(256),
                                                         torchvision.transforms.RandomCrop(224),
                                                         torchvision.transforms.RandomHorizontalFlip(),
                                                         torchvision.transforms.ToTensor(),
                                                         torchvision.transforms.Normalize([0.485, 0.456, -.406], [0.229, 0.224, 0.225])
                                                         ])
    data = MyDataset('/home/xykong/Friday_test/TrainAndValList/train.lst',transform=train_augmentation)#初始化类，设置数据集所在路径以及变换
    dataloader = DataLoader(data,batch_size=64,shuffle=True)#使用DataLoader加载数据

    model = load_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    running_loss = 0.0
    correct = 0
    for epoch in range(2):
        model.train()
        for i_batch,batch_data in tqdm(enumerate(dataloader)):


            inputs, labels = batch_data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            #print("图片大小：",inputs.shape) #torch.Size([128`, 3, 224, 224])
            #print("标签类别大小：", labels.shape)#torch.Size([128])
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):  # 当requires_grad设置为False时,反向传播时就不会自动求导了，因此大大节约了显存或者说内存。
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            preds = torch.max(outputs, 1)[1]
            loss.backward()
            optimizer.step()
            correct += torch.sum(preds == labels.data)
            # print statistics
            running_loss += loss.item()
            if i_batch % 10 == 9:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i_batch + 1, running_loss / 10))
                writer.add_scalar('train loss', running_loss/ 10, epoch)  # 第一个参数是名称，第二个参数是Y轴，第三个参数是X轴，类型是scalar，将其保存在文件里
                running_loss = 0.0
        print("train_acc=",correct.double() / len(dataloader.dataset))
        writer.add_scalar('train_acc', correct.double() / len(dataloader.dataset), epoch)
    print('Finished Training')
    PATH = './my_new_net.pth'
    torch.save(model.state_dict(), PATH)