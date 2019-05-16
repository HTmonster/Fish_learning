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
# fileName:train_textCNN_w2vec 
# project: Fish_learning
# author: theo_hui
# e-mail:Theo_hui@163.com
# purpose: {文件作用描述｝
# creatData:2019/5/16


import os
import datetime
import time
import tensorflow as tf
import numpy as np

from page_identify.TextCNN import TextCNN
from page_identify import data_Processer, word2vec_tool

#    参数设置
#================================================================

tf.flags.DEFINE_string("positive_url_file","./data/positive_urls.csv","正常URL数据集")
tf.flags.DEFINE_string("negative_url_file","./data/negative_urls.csv","恶意URL数据集")

#模型超参数
tf.flags.DEFINE_integer("embedding_size",10,"隐藏层的维度")
tf.flags.DEFINE_integer("max_seq_length",10,"输入序列的最大长度")
tf.flags.DEFINE_string("filter_sizes","3,4,5","卷积核(滤波器)的尺寸")
tf.flags.DEFINE_integer("num_filters",32,"卷积核的数目")
tf.flags.DEFINE_float("dropout_keep_prob",0.5,"DropOut层选择概率")
tf.flags.DEFINE_float("l2_reg_lambda",0.0,"l2正则化比例")
#tf.flags.DEFINE_boolean("use_glove",True,"是否使用GloVe模型")

#训练参数
tf.flags.DEFINE_integer("batch_size",500,"batch 大小")
tf.flags.DEFINE_integer("num_steps", 200, "训练的次数")
tf.flags.DEFINE_integer("evaluate_every", 100, "评价的间隔步数")
tf.flags.DEFINE_integer("checkpoint_every", 100, "保存模型的间隔步数")
tf.flags.DEFINE_integer("num_checkpoints", 5, "保存的checkpoints数")
tf.flags.DEFINE_float("validation_percentage", 0.2, "验证数据集比例")

# session配置的一些参数
tf.flags.DEFINE_boolean("allow_soft_placement", True, "允许tf自动分配设备")
tf.flags.DEFINE_boolean("log_device_placement", False, "日志记录")

#  解析参数
#=================================================================
# 解析参数
FLAGS = tf.flags.FLAGS
FLAGS.flag_values_dict()
print("\n*SETED FLAGS AS FOLLOW*\nFLAG_NAME\tFLAG_VALUE\n")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}\t{}".format(attr.upper(), value))
print("==========================================================================")

# 输出数据和模型的目录
# =======================================================
timestamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, "output/textCNN/runs",timestamp))
wVec_dir = os.path.abspath(os.path.join(os.path.curdir,"output/wordVec"))
print("\nWriting to {}\n".format(out_dir))
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
if not os.path.exists(wVec_dir):
    os.makedirs(wVec_dir)

#  加载数据
# =======================================================

print("\nLoading data...")
x_text, y = data_Processer.load_positive_negative_url_files_w2vec(FLAGS.positive_url_file,FLAGS.negative_url_file)
print(x_text)
print("total:",len(x_text))
print("\nloaded!")

# 数据规范化编码
#=========================================================

# 对齐
print("\n Loading embedding Layer tensor(padding)....")
x_padding, max_x_length = data_Processer.padding_url_w2vec(x_text,padding_url_length=FLAGS.max_seq_length)

# 映射编码 每个字符对应一个数字
print("padding done!")


print(x_padding[0])
x = np.array(word2vec_tool.embedding_sentences(x_padding, embedding_size = FLAGS.embedding_size, file_to_load = os.path.join(wVec_dir, 'trained_word2vec.model')))

#print(x)
print("x.shape = {}".format(x.shape))
print("y.shape = {}".format(y.shape))


# 数据处理,最终得到训练数据集
#=====================================================

# 随机打乱数据
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# 分隔验证和训练数据集
# TODO: This is very crude, should use cross-validation
dev_sample_index = -1 * int(FLAGS.validation_percentage * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

# 训练
# =======================================================
with tf.Graph().as_default():
    #配置会话
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess = tf.Session(config=session_conf)

    with sess.as_default():
        text_cnn = TextCNN(x_train.shape[1], #输入序列长度
                         y_train.shape[1], #分类数目
                         FLAGS.embedding_size,#隐藏层大小
                         list(map(int, FLAGS.filter_sizes.split(","))),#卷积核尺寸
                         FLAGS.num_filters,#卷积核数据
                         FLAGS.l2_reg_lambda)#l2正则化参数
        # 定义训练过程
        global_step = tf.Variable(0, name="global_step", trainable=False)  # 训练次数
        optimizer= tf.train.AdamOptimizer(1e-3)  # 优化算法
        grads_and_vars = optimizer.compute_gradients(text_cnn.loss)  # 计算相关的梯度
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)  # 运用梯度(gradients)

        # 追踪梯度值和稀疏值
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # 输出的路径
        print("Writing to {}\n".format(out_dir))

        # 正确率与损失率
        loss_summary = tf.summary.scalar("loss", text_cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", text_cnn.accuracy)

        # 训练总结
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # 验证总结
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # 存储检查点
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # 初始化所有变量
        sess.run(tf.global_variables_initializer())


        # 训练的一个步骤
        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
                text_cnn.input_x: x_batch,
                text_cnn.input_y: y_batch,
                text_cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, text_cnn.loss, text_cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)


        # 验证的一个步骤
        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
                text_cnn.input_x: x_batch,
                text_cnn.input_y: y_batch,
                text_cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, text_cnn.loss, text_cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)


        # 产生batch
        batches = data_Processer.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_steps)

        # 循环训练
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