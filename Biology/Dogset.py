from torchvision import datasets, transforms
import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import numpy as np
import codecs


IMG_SIZE=224

train_tf = transforms.Compose(
[lambda x:Image.open(x).convert("RGB"),
 transforms.RandomResizedCrop(IMG_SIZE),
 transforms.RandomHorizontalFlip(),
 transforms.ToTensor(),
 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225]),
 ]);
test_tf = transforms.Compose(
[lambda x:Image.open(x).convert("RGB"),
 transforms.Resize([IMG_SIZE,IMG_SIZE]),
 transforms.ToTensor(),
 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225]),
 ]);
val_tf = test_tf
class trainset(Dataset):
    def __init__(self, rootdir, imglst, phase):
        #定义好 image 的路径
        self.root = rootdir 
        self.cls2id = {}
        for name in sorted(os.listdir(os.path.join(rootdir))):
            # 过滤掉非目录文件
            if not os.path.isdir(os.path.join(rootdir, name)):
                continue
            self.cls2id[name] = int(name.strip().split("-")[1].split("n")[1])
        print(self.cls2id) #输出 类别-数字（class to id）
        filecp = codecs.open(imglst, encoding = 'cp1252')
        self.images = np.loadtxt(filecp,dtype=str,delimiter=',' )
        if phase == "train":
            self.tf = train_tf
        else:
            self.tf = val_tf

    def __getitem__(self, index):
        fn = self.images[index]
        fn = fn.split("//")[1]
        img = self.tf(os.path.join(self.root,fn)) # 图片存在的真实地址，然后tf（transform）得到裁剪等一些列变换的图片
        target = self.cls2id[fn.split("/")[0]]  #target 存类别
        return img, target #返回图片与标签

    def __len__(self):
        return len(self.images)


class testset(Dataset):
    def __init__(self, rootdir):
        #定义好 image 的路径
        self.root = rootdir 
        self.images = []
        self.names = []
        for filename in sorted(os.listdir(os.path.join(rootdir))):
            # 过滤掉目录文件
            if os.path.isdir(os.path.join(rootdir, filename)):
                continue
            self.images.append(os.path.join(rootdir,filename))
            self.names.append(filename)
            self.tf = test_tf

    def __getitem__(self, index):
        fn = self.images[index]
        img = self.tf(os.path.join(self.root,fn))
        return img, self.names[index]

    def __len__(self):
        return len(self.images)


def load_test(root_path,  batch_size):
    data = testset(root_path)
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=12)
    return data_loader

def load_train(root_path, imglst, batch_size, phase):
    data = trainset(root_path, imglst, phase)
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=12)
    return loader

