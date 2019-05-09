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
# fileName:train_simpleNN 
# project: Fish_learning
# author: theo_hui
# e-mail:Theo_hui@163.com
# purpose: 对simpleNN模型进行训练
# creatData:2019/5/9

import os
import datetime
import time
import tensorflow as tf
import numpy as np

from page_identify import simpleNN
from page_identify import data_Processer


#      参数设置
#============================================================

# 输入数据的一些参数
tf.flags.DEFINE_float("validation_percentage",0.1,"所有的训练数据用来验证的比例")
tf.flags.DEFINE_string("input_data_file", "./data/feature4.csv", "正样本数据")

#模型中的参数
tf.flags.DEFINE_integer("input_node", 8, "输入的维度") #8个特征
tf.flags.DEFINE_integer("embedding_node", 200, "隐藏层的维度")
tf.flags.DEFINE_integer("output_node", 2, "输出层的维度") #2分类
tf.flags.DEFINE_float("l2_reg_lambda", 0.0001, "L2 正则化比例")

# 训练的一些参数
tf.flags.DEFINE_float("learning_rate_base", 0.8, "基础学习率")
tf.flags.DEFINE_float("learning_rate_decay", 0.99, "学习率衰减率")
tf.flags.DEFINE_float("moving_average_decay", 0.99, "滑动平均衰减率")
tf.flags.DEFINE_integer("batch_size", 1000, "Batch大小")
tf.flags.DEFINE_integer("num_examples", 3800, "输入数据的数目")
tf.flags.DEFINE_integer("num_steps", 200, "训练的次数")
tf.flags.DEFINE_integer("evaluate_every", 100, "评价的间隔步数")
tf.flags.DEFINE_integer("checkpoint_every", 100, "保存模型的间隔步数")
tf.flags.DEFINE_integer("num_checkpoints", 5, "保存的checkpoints数")

# session配置的一些参数
tf.flags.DEFINE_boolean("allow_soft_placement", True, "允许tf自动分配设备")
tf.flags.DEFINE_boolean("log_device_placement", False, "日志记录")

# 解析参数
FLAGS = tf.flags.FLAGS
FLAGS.flag_values_dict()
print("\n*SETED FLAGS AS FOLLOW*\nFLAG_NAME\tFLAG_VALUE\n===========================================================")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}\t{}".format(attr.upper(), value))
print("==========================================================================")



# 设置输出的目录
#=====================================================
timestamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, "output",timestamp))
print("\n\nWriting to {}\n".format(out_dir))
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


# 加载数据
#=====================================================
print("\nLoading data...")
x, y_ = data_Processer.load_csv_file(FLAGS.input_data_file)
print(x,y_)
print("\nloaded!")

# 随机打乱数据
#======================================================
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y_)))
x_shuffled = x[shuffle_indices]
y_shuffled = y_[shuffle_indices]

# 分隔验证和训练数据集
# TODO: This is very crude, should use cross-validation
dev_sample_index = -1 * int(FLAGS.validation_percentage * float(len(y_)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
print("\nTrain/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))



# 开始训练
#=========================================================
with tf.Graph().as_default():

    #参数配置
    session_conf = tf.ConfigProto(
        allow_soft_placement = FLAGS.allow_soft_placement,#如果指定的设备不存在 是否允许tf自动分配
	log_device_placement = FLAGS.log_device_placement)    #是否打印设备分配日志

    #创建会话
    sess = tf.Session(config = session_conf)
    with sess.as_default():

        #建立神经网络
        simple_nn=simpleNN.simpleNN(FLAGS.input_node,FLAGS.embedding_node,FLAGS.output_node,FLAGS.l2_reg_lambda)

        #定义训练过程
        global_step = tf.Variable(0, name="global_step", trainable=False)#训练次数
        # 滑动平均
        variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay, global_step)  # 初始滑动平均类
        variable_averages_op = variable_averages.apply(tf.trainable_variables())  # 所有变量使用滑动平均类

        # 指数衰减的学习率
        learning_rate = tf.train.exponential_decay(FLAGS.learning_rate_base,global_step,FLAGS.num_examples /FLAGS.batch_size,FLAGS.learning_rate_decay)
        # 使用优化算法来优化损失函数
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(simple_nn.loss, global_step=global_step)

        #使用滑动平均来应用于训练步骤
        with tf.control_dependencies([train_step, variable_averages_op]):
            train_op = tf.no_op(name='train')

        #正确率与损失率
        loss_summary = tf.summary.scalar("loss", simple_nn.loss)
        acc_summary = tf.summary.scalar("accuracy", simple_nn.accuracy)

        #训练总结
        train_summary_op = tf.summary.merge([loss_summary, acc_summary])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        #验证总结
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        #存储检查点
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        #初始化所有变量
        sess.run(tf.global_variables_initializer())

        #训练的一个步骤
        def train_step(x_batch, y_batch):
            feed_dict = {
              simple_nn.input_x: x_batch,
              simple_nn.input_y: y_batch,
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, simple_nn.loss, simple_nn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        #验证的一个步骤
        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              simple_nn.input_x: x_batch,
              simple_nn.input_y: y_batch,
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, simple_nn.loss, simple_nn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        #产生batch
        batches = data_Processer.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_steps)

        #循环训练
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
