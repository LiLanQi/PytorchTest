# coding='utf-8'
"""t-SNE对手写数字进行可视化"""
from time import time

import numpy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.manifold import TSNE
from sklearn import datasets
import seaborn as sns


def get_data():
    digits = datasets.load_digits(n_class=6)
    data = digits.data
    label = digits.target
    n_samples, n_features = data.shape
    return data, label, n_samples, n_features


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        # plt.scatter(data[i, 0], data[i, 1], c=plt.cm.Set1(label[i] / 10.), cmap=plt.cm.Spectral)
        plt.scatter(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i]),
                 )
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig


def main():
    data, label, n_samples, n_features = get_data()
    print("data=", data)
    print("data.shape=", data.shape)
    print("label=", label)
    print("label.shape=", label.shape)
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    result = tsne.fit_transform(data)
    print("result.shape=", result.shape)
    fig = plot_embedding(result, label,
                         't-SNE embedding of the digits (time %.2fs)'
                         % (time() - t0))
    plt.show(fig)


if __name__ == '__main__':
    # main()
    # df = pd.read_csv('D:/set2set.csv')
    # init_data = torch.zeros((3,3))
    # print(len(df))
    # print(df.loc[18])

    data = np.loadtxt('D:/solution_data.csv',delimiter=",")
    tsne = TSNE(n_components=2, init='pca')
    batch_size = 192
    t0 = time()
    label_index = []
    for iter in range(12):
        for i in range(5):
            for j in range(batch_size):
                label_index.append(i)
    # data = np.loadtxt('D:/solution.csv', delimiter=",")
    # tsne = TSNE(n_components=2, init='pca')
    # batch_size = 192
    # t0 = time()
    # label_index = []
    # for iter in range(1):
    #     for i in range(5):
    #         for j in range(batch_size):
    #             label_index.append(i)

    label = pd.Series(label_index)
    print(label)
    # print("result.shape=", result.shape)
    # print(result)
    # print("result=",result)
    label = label.map({0:"ACE", 1:"NMF", 2:"DMF", 3:"Water", 4:"Meth"})
    # print(label.shape)
    label = pd.DataFrame(label).T
    # print(label)
    result = tsne.fit_transform(data)

    result = pd.DataFrame(result.T)
    # result = [reslut, label]
    result = pd.concat((result, label)).T

    # result = numpy.vstack((result, label))
    # result = torch.cat((torch.tensor(result), label), 1)
    # print(result)
    df_tsne = pd.DataFrame(result.values, columns=["X", "Y", "solution"])
    print(df_tsne)
    # print(df_tsne)
    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=df_tsne, x = "X", y="Y", hue="solution")
    plt.show()
