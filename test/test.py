import numpy 
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

#labels_dense = numpy.arange(20)
#index_offset = geek.arange(20) * 13
#print(labels_dense)
#labels_one_hot = geek.zeros((20, 13))
# print(labels_one_hot)
# print(labels_one_hot.ravel())
#print(index_offset+labels_dense.ravel())
#x=numpy.zeros(shape=(3,3))
#print(x)


def ravel():  
  a=numpy.arange(20)
  #1,2,3,4
  print("a:",a)
  a.flat[[1,2,1000]]=100
  print("x:",a)


def truncated_normal():
    c=tf.truncated_normal([5,5,1,32],stddev=0.1)
    with tf.Session() as sess:
        print(sess.run(c))

#truncated_normal()

def constant():
    c=tf.constant(0.1,shape=[32,32])
    
    with tf.Session() as sess:
        print(sess.run(c))

constant()

def reshape():
    c=tf.reshape([1,2,3,4],[-1,2,2,1])
    with tf.Session() as sess:
        print(sess.run(c))

#reshape()


def conv2d():
  x=tf.Variable(tf.reshape([[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0]],[-1,4,4,1]))
  W=tf.Variable(tf.truncated_normal([5,5,1,10], stddev=0.1))
  c=tf.nn.conv2d(x,W, strides=[1, 1, 1, 1], padding='SAME')
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(c))

#conv2d()