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


def load_csv_file(input_file,cols=9):
    '''
    加载特征和标签
    :param input_file: csv文件  默认最后一行为标签
    :return:
    '''

    #加载文件
    csv_data=pd.read_csv(input_file,usecols=[i for i in range(1,cols)])
    csv_label=pd.read_csv(input_file,usecols=[cols])

    #label 转换 0-> 1,0  1->0,1
    label_extend=[]
    for label in np.array(csv_label):
        if label== 0:
            label_extend.append([1,0])
        else:
            label_extend.append([0,1])
    label_extend=np.array(label_extend,dtype=int)

    #转换数据
    data=np.float32(csv_data)
    label=np.array(label_extend)

    #返回数据
    return (data,label)

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

    print(load_csv_file("./data/feature4.csv"))