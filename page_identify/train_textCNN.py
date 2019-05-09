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
# fileName:train_textCNN 
# project: Fish_learning
# author: theo_hui
# e-mail:Theo_hui@163.com
# purpose: 对TextCNN模型的训练
# creatData:2019/5/9
import os
from datetime import time
import time
import tensorflow as tf

from page_identify import data_Processer


#      参数设置
#============================================================

# 输入输出数据的一些参数
tf.flags.DEFINE_float("validation_percentage",0.1,"所有的训练数据用来验证的比例")
tf.flags.DEFINE_string("positive_data_file", "./data/ham_100.utf8", "正样本数据")
tf.flags.DEFINE_string("negative_data_file", "./data/spam_100.utf8", "负样本数据")
tf.flags.DEFINE_integer("num_labels",2,"要分类的数目 二分类")

# textCNN中的参数
tf.flags.DEFINE_integer("embedding_dim", 128, "隐藏层的维度")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "每个过滤器的尺寸")
tf.flags.DEFINE_integer("num_filters", 128, "每个过滤器尺寸的过滤器数量")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout 比例")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 正则化比例")

# 训练的一些参数
tf.flags.DEFINE_integer("batch_size", 64, "Batch大小")
tf.flags.DEFINE_integer("num_steps", 200, "训练的次数")
tf.flags.DEFINE_integer("evaluate_every", 100, "评价的间隔步数")
tf.flags.DEFINE_integer("checkpoint_every", 100, "保存模型的间隔步数")
tf.flags.DEFINE_integer("num_checkpoints", 5, "保存的checkpoints数")

# session配置的一些参数
tf.flags.DEFINE_boolean("allow_soft_placement", True, "允许tf自动分配设备")
tf.flags.DEFINE_boolean("log_device_placement", False, "日志记录")


#解析参数
#=======================================================================

FLAGS = tf.flags.FLAGS
FLAGS.flag_values_dict()
print("\n*SETED FLAGS AS FOLLOW*\nFLAG_NAME\tFLAG_VALUE\n"
      "===========================================================")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}\t{}".format(attr.upper(), value))

#输出目录设置
#=======================================================================
timestamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, "output",timestamp))

print("\nWriting to {}\n".format(out_dir))
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

#加载数据
#========================================================================
print("\nLoading data...")
x_text, y =data_Processer.load_csv_file("./data/feature4.csv")
print(x_text)
print(y)
print("\nloaded!")

#获得隐藏层向量
#=======================================================================




