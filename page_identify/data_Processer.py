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
import copy
import re
import string

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

def clean_URL(url_str):
    '''
    清理URL
    :param url_str: 处理之前的URL
    :return: 处理之后的URL
    '''
    #先将字节数组转换为字符
    url_str=str(url_str)

    #去掉常见的字符
    url_str=re.sub(r"http://","",url_str)
    url_str=re.sub(r"https://","",url_str)
    url_str=re.sub(r"www\.","",url_str)
    url_str=re.sub(r"/$"," ",url_str)

    return url_str

def load_positive_negative_url_files(positive_url_file,negative_url_file):
    '''
    加载恶意的还有正常的URL并进行清理 混合处理
    :param positive_url_file:
    :param negative_url_file:
    :return:
    '''

    #从CSV文件中读取数据
    positive_url_data = pd.read_csv(positive_url_file,usecols=[0]).values
    negative_url_data = pd.read_csv(negative_url_file,usecols=[0]).values

    # print(positive_url_data)
    # print(negative_url_data)

    #对数据进行简单的清洗
    positive_clean_url_data=[clean_URL(url_data[0]) for url_data in positive_url_data]
    negative_clean_url_data=[clean_URL(url_data[0]) for url_data in negative_url_data]


    # print(positive_clean_url_data)
    # print(negative_clean_url_data)

    # 将数据进行结合
    x_text = positive_clean_url_data + negative_clean_url_data

    # 产生标签
    positive_labels = [[0, 1] for _ in positive_clean_url_data]
    negative_labels = [[1, 0] for _ in negative_clean_url_data]
    y = np.concatenate([positive_labels, negative_labels], 0)

    return [x_text,y]


def padding_url(urls,padding_token='0', padding_url_length = None):
    '''
    对URL集合进行对齐处理
    :param url:要处理的url集合
    :return:
    '''

    #最大长度
    max_url_length=padding_url_length if padding_url_length else max([len(url) for url in urls])

    print("\n********padding url {} *******\n".format(max_url_length))

    #对齐处理
    for i in range(len(urls)):
        url=urls[i]

        if len(url)>max_url_length:
            urls[i]=url[:max_url_length]#截断
        else:
            urls[i]=url+padding_token*(max_url_length-len(url))#填充

    return (urls,max_url_length)



def map_code_char(urls,max_sequece_length):
    '''
    对URL字符进行字符级别的映射编码处理
    :param urls: url集合
    :param max_sequece_length: 最大的字符集合
    :return: 每个URL为一个一维向量
    '''
    # 字符对应表
    characters = string.printable  # 返回字符串，所有可打印的 ASCII 字符
    token_index = dict(zip(range(1, len(characters) + 1), characters))  # 给字符添加索引
    num_characters = max(token_index.keys())

    print("\n-------------------编码对应表--------------------\n")
    print(token_index)

    # 二维矩阵
    embed_matrix = np.zeros((len(urls), max_sequece_length))

    # 对每个url进行处理
    for url in urls:
        for i, sample in enumerate(url):
            for j, character in enumerate(sample):
                index = token_index.get(character)
                embed_matrix[i, j] = index
    return embed_matrix

def one_hot_char(urls,max_sequence_length):
    '''
    对URL进行字符级别的one-hot处理

    :param urls: url集合
    :param embedding_size: 后面嵌入层大小
    :return: 每一个URL输出为一个二维矩阵
    '''

    #字符对应表
    characters=string.printable#返回字符串，所有可打印的 ASCII 字符
    token_index=dict(zip(range(1,len(characters)+1),characters))#给字符添加索引
    num_characters=max(token_index.keys())

    print("\n-------------------编码对应表--------------------\n")
    print(token_index)

    embed_matrix = np.zeros((len(urls), max_sequence_length, num_characters))

    #对每个url进行处理
    for url in urls:
        for i,sample in enumerate(url):
            for j,character in enumerate(sample):
                index=token_index.get(character)
                embed_matrix[i,j,index]=1.
    return (embed_matrix,num_characters)

def clean_split_url_w2vec(url_str):
    # 先将字节数组转换为字符
    url_str = str(url_str)

    # 去掉常见的字符
    url_str = re.sub(r"http://", "", url_str)
    url_str = re.sub(r"https://", "", url_str)
    url_str = re.sub(r".html$", "", url_str)
    url_str = re.sub(r".htm$", "", url_str)

    #分隔字符
    url_str=re.split(r"[/=-?.&]",url_str)

    return url_str



def load_positive_negative_url_files_w2vec(positive_url_file,negative_url_file):
    '''
    加载恶意的还有正常的URL  word2vec方案 （对单词进行处理）
    :param positive_url_file:
    :param negative_url_file:
    :return:
    '''

    #从CSV文件中读取数据
    positive_url_data = pd.read_csv(positive_url_file,usecols=[0]).values
    negative_url_data = pd.read_csv(negative_url_file,usecols=[0]).values

    print(positive_url_data)
    print(negative_url_data)

    #对数据进行简单的清洗
    positive_clean_split_url_data=[clean_split_url_w2vec(url_data[0]) for url_data in positive_url_data]
    negative_clean_split_url_data=[clean_split_url_w2vec(url_data[0]) for url_data in negative_url_data]


    print(positive_clean_split_url_data)
    print(negative_clean_split_url_data)

    # 将数据进行结合
    x_text = positive_clean_split_url_data + negative_clean_split_url_data

    # 产生标签
    positive_labels = [[0, 1] for _ in positive_clean_split_url_data]
    negative_labels = [[1, 0] for _ in negative_clean_split_url_data]
    y = np.concatenate([positive_labels, negative_labels], 0)

    return [x_text,y]

def padding_url_w2vec(urls,padding_token="<PADDING>", padding_url_length = None):
    '''
    对URL集合进行对齐处理
    :param url:要处理的url集合
    :return:
    '''

    #最大长度
    max_url_length=padding_url_length if padding_url_length else max([len(url) for url in urls])

    print("\n********padding url {} *******\n".format(max_url_length))

    #对齐处理
    padding_urls=[]
    for url in urls:
        if len(url)>max_url_length:
            url=url[:max_url_length]
            padding_urls.append(copy.deepcopy(url))
        else:
            url.extend([padding_token for _ in range(max_url_length-len(url))])
            padding_urls.append(url)


    return (padding_urls,max_url_length)


if __name__ == '__main__':

    # print(load_csv_file("./data/NegativeFile6.csv"))
    # print(load_csv_file("./data/PositiveFile6.csv"))
    #print(load_positive_negative_url_files_w2vec("./data/PositiveFile6.csv","./data/NegativeFile6.csv"))

    #load_url_csv_file("./data/normalURL/Arts.csv")
    load_positive_negative_url_files_w2vec("./data/positive_urls.csv","./data/negative_urls.csv")
    print("he")