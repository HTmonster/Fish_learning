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
# fileName:TextCNN 
# project: TextCNN_SpamSort
# author: theo_hui
# e-mail:Theo_hui@163.com
# purpose: CNN模型 采用Yoon Kim 提出的TextCNN模型
# creatData:2019/5/9

import tensorflow as tf

class TextCNN():
    '''
     文本分类的CNN模型

     1. Embeddding Layer  隐藏层
     2. Convolution Layer 卷积层
     3. Max-Poling Layer  最大值池化层
     4. Softmax Layer     softmax层
    '''

    def __init__(self,
                 sequence_length, #序列长度 输入定长处理 超过的截断 不足的补零
                 num_classes,     #分类的数目 分为几类
                 embedding_size,  #词向量的维度 （降维）
                 filter_sizes,    #所有过滤器的尺寸
                 num_filters,     #过滤器的数量
                 l2_reg_lambda=0.0):

        #输入
        self.input_x=tf.placeholder(tf.float32,[None,sequence_length,embedding_size],name="input_x") #输入数据
        self.input_y=tf.placeholder(tf.float32,[None,num_classes],name="input_y")                    #验证数据
        self.dropout_keep_prob=tf.placeholder(tf.float32,name="dropout_keep_prob")                   #dropout


        #隐藏层
        self.EmbeddinfLayer()

        #卷积层+max_pool层
        self.Convoluthion_maxpoolLayer(filter_sizes,embedding_size,num_filters,sequence_length)

        #dropout层 (防止过拟合)
        self.DropoutLayer()

        #输出层
        self.OutputLayer(num_filters*len(filter_sizes),num_classes)

        #计算损失函数 交叉熵
        self.calc_loss(l2_reg_lambda)

        #计算准确率
        self.calc_accuracy()



    def EmbeddinfLayer(self):
        '''
        隐藏层

        将one-hot编码的词投影到一个低维的空间中
        '''

        self.embedded_chars=self.input_x
        self.embedded_chars_expended=tf.expand_dims(self.embedded_chars,-1)#增加维度

    def Convoluthion_maxpoolLayer(self,filter_sizes,embedding_size,num_filters,sequence_length):
        '''
        卷积层+maxpool层

        为不同尺寸的filter都建立一个卷积层（多个feature map)
        '''

        #循环遍历建立
        pooled_outputs=[]
        for i,filter_size in enumerate(filter_sizes):

            #不同的命名空间
            with tf.name_scope("conv-maxpool-%s"%filter_size):
                #卷积层
                filter_shape=[filter_size,embedding_size,1,num_filters]
                weight=tf.Variable(tf.truncated_normal(filter_shape,stddev=0.1),name="weight")
                bias =tf.Variable(tf.constant(0.1,shape=[num_filters]),name="biase")

                conv=tf.nn.conv2d(self.embedded_chars_expended,#input 输入数据 Tensor
                                  weight,                      #filter 卷积核 Tensor 其shape必须为[高度,宽度,通道数,个数]
                                  strides=[1,1,1,1],           #strides 每一维的步长
                                  padding="VALID",             #padding  "SAME"或者"VALID"之一 决定不同的卷积方式
                                  name="conv")

                #relu激活函数（非线性）
                relued=tf.nn.relu(tf.nn.bias_add(conv,bias),#features 将卷积加上bias
                             name="relu")

                #Maxpooling 池化
                pooled=tf.nn.max_pool(relued,    #value 需要池化的的输入 [batch, height, width, channels]这样的shape
                                      ksize=[1,sequence_length-filter_size+1,1,1], #池化窗口大小 四维向量 一般是[1, height, width, 1]，不想在batch和channels上做池化
                                      strides=[1,1,1,1], #每一维上的步长
                                      padding='VALID', #padding  "SAME"或者"VALID"之一 决定不同的卷积方式
                                      name="pool")

                #添加到集合中
                pooled_outputs.append(pooled)

        #结合所有池化后的特征
        num_filter_total=num_filters*len(filter_sizes)

        self.h_pool = tf.concat(pooled_outputs,3) #张量连接 维度3
        self.h_pool_flat = tf.reshape(self.h_pool,[-1,num_filter_total])


    def DropoutLayer(self):
        '''
        Dropout层

        防止过拟合
        '''

        with tf.name_scope("dropout"):
            self.h_drop=tf.nn.dropout(self.h_pool_flat, #数据
                                      self.dropout_keep_prob  #dropout 概率
                                      )
    def OutputLayer(self,num_filters_total,num_classes):
        '''
        输出层
        '''

        #L2正则化损失
        self.l2_loss=tf.constant(0.0)

        with tf.name_scope("output"):
            weight=tf.get_variable("weight",
                                   shape=[num_filters_total,num_classes],
                                   initializer=tf.contrib.layers.xavier_initializer()#保持每一层梯度大小差不多
                                   )
            bias=tf.Variable(tf.constant(0.1,shape=[num_classes],name="bias"))

            self.l2_loss+=tf.nn.l2_loss(weight)
            self.l2_loss+=tf.nn.l2_loss(bias)

            #得到结果
            self.score=tf.nn.xw_plus_b(self.h_drop,weight,bias,name="score")#相当与matmul(x,weigt)+bias
            self.predictions=tf.argmax(self.score,1,name="predictions")


    def calc_loss(self,l2_reg_lambda):
        '''
        计算损失函数

        使用交叉熵来计算
        '''

        with tf.name_scope("loss"):

            losses=tf.nn.softmax_cross_entropy_with_logits(logits=self.score,labels=self.input_y)
            self.loss=tf.reduce_mean(losses)+l2_reg_lambda*self.l2_loss #总损失

    def calc_accuracy(self):
        '''
        计算准确率
        '''

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions,tf.argmax(self.input_y,1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions,"float"),name="accuracy")