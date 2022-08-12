# 引入tensorflow1.0并禁用v2
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

#权重在初始化时应该加入少量的噪声来打破对称性以及避免0梯度，避免神经元节点输出恒为0的问题（dead neurons）
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

#生成shape维度的0.1
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#二维卷积
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#最大池化，卷积算法中临近点的最大值
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

#测试