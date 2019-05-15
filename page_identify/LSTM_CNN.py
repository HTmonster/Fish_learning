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
# fileName:LSTM_CNN 
# project: Fish_learning
# author: theo_hui
# e-mail:Theo_hui@163.com
# purpose: LSTM_CNN模型
#          LSTM_CNN模型兼具了RNN和CNN的优点，考虑了句子的序列信息，又能捕捉关键信息 但是 没法并行计算，因此，训练时速度要比FastText和TextCNN慢得多
# creatData:2019/5/10
import tensorflow as tf


class LSTM_CNN(object):
    # 1. Embed --> LSTM
    # 2. LSTM --> CNN
    # 3. CNN --> Pooling/Output

    def __init__(self,
                 sequence_length, #输入序列长度
                 num_classes,     #分类数目
                 vocab_size,      #词汇尺寸
                 embedding_size,  #隐藏层尺寸
                 filter_sizes,    #过滤器(卷积核)尺寸
                 num_filters,     #过滤器(卷积核)数量
                 l2_reg_lambda=0.0,#l2正则化参数
                 ):
        #L2正则化
        self.l2_loss = tf.constant(0.0)

        #输入训练数据与验证数据以及dropout层
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")  # X - The Data
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")  # Y - The Lables
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")  # Dropout

        #隐藏层
        self.EmbeddingLayer(vocab_size,embedding_size)

        #LSTM层
        self.LSTMLayer()

        #卷积层+maxpool层
        num_filters_total=self.ConvMaxpoolLayer(sequence_length,embedding_size,filter_sizes,num_filters)

        # Dropout层
        self.DropoutLayer()

        # 输出层 得分和预测
        self.OutputLayer(num_filters_total,num_classes)

        # loss
        self.calc_loss(l2_reg_lambda)

        # accuracy
        self.calc_acc()


#-------------------------------------------------------------------------------------

    def EmbeddingLayer(self,vocab_size,embedding_size):
        '''
        隐藏层
        :return:
        '''
        with tf.name_scope("embedding"):
            self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x) #embedding_lookup 选取一个张量里面对应的索引元素
            # self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

    def LSTMLayer(self):
        '''
        LSTM层
        :return:
        '''
        self.lstm_cell = tf.contrib.rnn.LSTMCell(32, state_is_tuple=True)
        # self.h_drop_exp = tf.expand_dims(self.h_drop,-1)
        self.lstm_out, self.lstm_state = tf.nn.dynamic_rnn(self.lstm_cell, self.embedded_chars, dtype=tf.float32)

        self.lstm_out_expanded = tf.expand_dims(self.lstm_out, -1)

    def ConvMaxpoolLayer(self,sequence_length,embedding_size,filter_sizes,num_filters):
        '''
        卷积层+maxpool层
        :return:
        '''
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                #卷积层 以LSTM层的输出为输入
                filter_shape = [filter_size, embedding_size, 1, num_filters]

                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")

                conv = tf.nn.conv2d(self.lstm_out_expanded, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")

                #非线性化
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

                #每一个卷积层的池化层
                pooled = tf.nn.max_pool(h, ksize=[1, sequence_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1],
                                        padding='VALID', name="pool")
                pooled_outputs.append(pooled)

        # 结合所有的池化层
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        return num_filters_total

    def DropoutLayer(self):
        '''
        Dropout层
        :return:
        '''
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

    def OutputLayer(self,num_filters_total,num_classes):
        '''
        输出层 并计算L2正则化损失 得分和预测
        :param num_filters_total: 卷积层的所有大小
        :param num_classes: 分类数目
        :return:
        '''

        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")

            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

    def calc_loss(self,l2_reg_lambda):
        '''
        计算损失率
        :return:
        '''
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * self.l2_loss

    def calc_acc(self):
        '''
        计算正确率
        :return:
        '''
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
