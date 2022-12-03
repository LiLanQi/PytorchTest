import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms,utils
import numpy as np
# 使用ImageFolder需要保证数据集以下列形式组织：
'''
root/dog/xxx.png
root/dog/xxy.png
root/dog/xxz.png

root/cat/123.png
root/cat/nsdf3.png
root/cat/asd932_.png
'''
img_data = torchvision.datasets.ImageFolder(
    root = r'data/low-resolution',
    transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()])
        )
print('数据集类别：',img_data.classes)
print('数据集类别大小：',len(img_data.classes))
print('数据集大小：',len(img_data))

# 使用torch.utils.data.DataLoader加载，形成一个DataLoader类实例
data_loader = torch.utils.data.DataLoader(img_data,batch_size=36, shuffle=True)
print(len(data_loader))

def imshow(img):
#    img = img / 2 + 0.5     # unnormalize
    img = torchvision.utils.make_grid(img, nrow=6)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title('Batch from dataloader')
    plt.xticks([])
    plt.yticks([])
    plt.show()

# get some random training images
dataiter = iter(data_loader)
images, labels = dataiter.next()
print(images.shape, labels)
for lable in labels:
    print(img_data.classes[lable])

# show images
imshow(images)

