#!/usr/bin/python3
# -*- coding:utf-8 -*-
#   ____    ____    ______________
#  |    |  |    |  |              |
#  |    |  |    |  |_____    _____|
#  |    |__|    |       |    |
#  |     __     |       |    |
#  |    |  |    |       |    |
#  |    |  |    |       |    |
#  |____|  |____|       |____|
#
# fileName:data_Processer 
# project: Fish_learning
# author: theo_hui
# e-mail:Theo_hui@163.com
# purpose: 对数据进行处理
# creatData:2019/5/9

import tensorflow as tf
import numpy as np
import pandas as  pd


def load_csv_file(input_file,cols=7):
    '''
    加载特征
    :param input_file: csv文件  默认第一行为文件名 要去除
           cols 行数
    :return:
    '''

    #加载文件
    csv_data=pd.read_csv(input_file,usecols=[i for i in range(1,cols)])

    #转换数据
    data=np.float32(csv_data)

    #返回数据
    return data

def load_positive_negtive_data_files(positive_file,negative_file):
    '''
    加载正向数据和负向数据  并添加标签混合
    :param positive_file: 正向数据
    :param negative_file: 负向数据
    :return:
    '''

    positive_examples=load_csv_file(positive_file)
    negative_examples=load_csv_file(negative_file)

    #生成标签
    positive_labels=[[0,1] for _ in positive_examples]
    negative_labels=[[1,0] for _ in negative_examples]

    #分别混合数据与标签
    x=np.concatenate([positive_examples,negative_examples],0)
    y=np.concatenate([positive_labels,negative_labels],0)

    return (x,y)

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    '''
    产生数据batch的迭代
    '''
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        if shuffle:
            # Shuffle the data at each epoch
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_idx : end_idx]

if __name__ == '__main__':

    print(load_csv_file("./data/NegativeFile6.csv"))
    print(load_csv_file("./data/PositiveFile6.csv"))
    print(load_positive_negtive_data_files("./data/PositiveFile6.csv","./data/NegativeFile6.csv"))