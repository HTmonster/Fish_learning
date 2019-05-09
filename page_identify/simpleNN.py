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
# fileName:simpleNN 
# project: Fish_learning
# author: theo_hui
# e-mail:Theo_hui@163.com
# purpose: 简单的神经网络
# creatData:2019/5/9

import tensorflow as tf

class simpleNN():
    '''
    简单的神经网络

    输入层
    隐藏层 (1层)
    输出层
    '''

    # 辅助函数 获取变量 使用 get_Variable来获取
    def get_weight_variable(self,shape, regularizer=None):
        # 获取一个变量 名字为weights 形状由参数指定
        weights = tf.get_variable(
            "weight", shape=shape, initializer=tf.truncated_normal_initializer(stddev=0.1))

        # 当给出正则化损失函数时候
        # 将当前变量的正则化损失加入到losses集合

        if regularizer != None:
            tf.add_to_collection('losses', regularizer(weights))

        return weights

    #构造函数
    def __init__(self,input_node,embedding_node,output_node,l2_reg_lambda):

        # 输入数据与验证的placeholder
        self.input_x = tf.placeholder(tf.float32, [None,input_node], name="x-input")
        self.input_y = tf.placeholder(tf.float32, [None,output_node], name="y-input")

        # 正则化函数
        regularizer = tf.contrib.layers.l2_regularizer(l2_reg_lambda)

        # 输入层到隐藏层 input->embed
        with tf.variable_scope('layer1'):
            weights = self.get_weight_variable([input_node,embedding_node], regularizer)  # 权重
            biases = tf.get_variable("biases", [embedding_node], initializer=tf.constant_initializer(0.0))  # 偏置项

            # 向前传播
            self.layer1 = tf.nn.relu(tf.matmul(self.input_x, weights) + biases)

        # 隐藏层到输出层 embed->output
        with tf.variable_scope('layer2'):
            weights = self.get_weight_variable([embedding_node, output_node], regularizer)
            biases = tf.get_variable("biases", [output_node], initializer=tf.constant_initializer(0.0))
            self.y = tf.matmul(self.layer1, weights) + biases
            self.predictions = tf.argmax(self.y, 1, name="predictions")

        # 损失函数 使用交叉熵
        with tf.name_scope("loss"):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.y, labels=tf.arg_max(self.input_y, 1))
            cross_entropy_mean = tf.reduce_mean(cross_entropy)

            # 总损失函数 交叉熵+L2正则化损失函数
            self.loss = cross_entropy_mean + tf.add_n(tf.get_collection("losses"))

        #正确率
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")