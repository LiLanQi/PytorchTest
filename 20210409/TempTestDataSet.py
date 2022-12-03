import numpy as np
import torchvision
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from MyDataset import MyDataset
from Net import Net
def load_model():
    model = torchvision.models.vgg19(pretrained=True)
    n_features = model.classifier[6].in_features
    fc = torch.nn.Linear(n_features, 131)
    model.classifier[6] = fc
    model.classifier[6].weight.data.normal_(0,0.005)
    model.classifier[6].bias.data.fill_(0.1)
    return model

if __name__=='__main__':
    train_augmentation = torchvision.transforms.Compose([torchvision.transforms.Resize(256),
                                                         torchvision.transforms.RandomCrop(224),
                                                         torchvision.transforms.RandomHorizontalFlip(),
                                                         torchvision.transforms.ToTensor(),
                                                         torchvision.transforms.Normalize([0.485, 0.456, -.406], [0.229, 0.224, 0.225])
                                                         ])

    val_data = MyDataset('/home/xykong/Friday_test/TrainAndValList/validation.lst', transform=train_augmentation)  # 初始化类，设置数据集所在路径以及变换
    val_dataloader = DataLoader(val_data, batch_size=64, shuffle=True)  # 使用DataLoader加载数据
    torch.manual_seed(10)
    DEVICE = torch.device('cuda:0')

    model = load_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)



    # net.cuda()
    PATH = './my_new_net.pth'
    model.load_state_dict(torch.load(PATH))


    class_correct = list(0. for i in range(131))
    class_total = list(0. for i in range(131))
    correct = 0
    total = 0
    with torch.no_grad():
        for data in val_dataloader:
            images = data[0].to(DEVICE)
            labels = data[1].to(DEVICE)
          #  print("type(images)=",type(images), "type(labels)=",type(labels),"type(net)=",type(net),"data[0].cuda()",type(data[0].cuda()))
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(1,130):
        print('Accuracy of %5s : %2d %%' % (
            i, 100 * class_correct[i] / class_total[i]))

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))

