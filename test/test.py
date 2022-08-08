import sys
import numpy
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def ravel():
    a = numpy.arange(20)
    # 1,2,3,4
    print("a:", a)
    a.flat[[1, 2, 1000]] = 100
    print("x:", a)


def truncated_normal():
    c = tf.truncated_normal([5, 5, 1, 10], stddev=0.1)
    with tf.Session() as sess:
        print(sess.run(c))


def constant():
    c = tf.constant(0.1, shape=[32, 32])

    with tf.Session() as sess:
        print(sess.run(c))


def reshape():
    c = tf.reshape([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                    9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]], [-1, 4, 4, 1])
    with tf.Session() as sess:
        print(sess.run(c))


def conv2d():
    x = tf.Variable(tf.reshape([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                    9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]], [-1, 4, 4, 1]))
    W = tf.Variable([[[[2.0]],[[2.0]]],[[[2.0]],[[2.0]]]])

    c = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    #d = tf.nn.max_pool(c, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(c))


if __name__ == '__main__':
    method = sys.argv[1]
    if method == 'conv2d':
        conv2d()
    elif method == 'reshape':
        reshape()
    elif method == 'constant':
        constant()
    elif method == 'truncated_normal':
        truncated_normal()
    elif method == 'ravel':
        ravel()
