import PIL
import torch
from skimage import io
import os
from PIL import Image
from torch.utils.data import Dataset

class MyDataset(Dataset):

    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform):
        self.root_dir = root_dir   #文件目录
        self.transform = transform #变换
        file1 = open(root_dir, "r", encoding='UTF-8') #root_dir = D:/DeepLearning/PytorchTest/data/train.lst
        file_list1 = file1.readlines()
        path_list = self.readFile(file_list1)
        self.images = path_list

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]  # 根据索引index获取该图片绝对地址.D:/DeepLearning/PytorchTest/data/806-n000129-papillon/n170760.jpeg
        #print(image_path)
        img = Image.open(image_path).convert("RGB")  # 读取该图片

        img = self.transform(img)
        label = image_path.split('/')[-2].split('-')[-2][1:]
        #sample = {'image': img, 'label': label}  # 根据图片和标签创建字典
        label = int(label)
        return img,label  # 返回该样本

    def readFile(self, file_list1):
        path_list = []
        for i in range(1,len(file_list1)):
            temp_path = file_list1[i][3:]
            path = os.path.join("data/low-resolution/", temp_path.strip())
            #print("1:",file_list1[i],',2:',temp_path,",3:",path)
            path_list.append(path)
        return path_list
