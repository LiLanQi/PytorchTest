import numpy as np
from sklearn.model_selection import train_test_split
import torch
datas = []
atom_datas = str.split("a,b,c,d,e",",")
count = 0
# for atom in atom_datas:
#     print("count=",count,"atom=",atom)
#     count = count+1
data_list = []
for i in range(10):
    data_list.append(i)

# X_train, X_test = train_test_split(data_list, train_size=0.7, random_state=42)
# print("X_train=",X_train)
# for i in range(len(X_train)):
#     print(X_train[i])
list1 = (1000,2,3,4)
list2 = [4321,3123,2321,1123]
outputs = torch.tensor(list1).squeeze(-1)
lables = torch.tensor(list2)
print(outputs.shape)
print(outputs.shape[0])