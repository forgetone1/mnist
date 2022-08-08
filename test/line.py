import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def line():
    x = tf.constant([0.9, 0.85], shape=[1, 2], name="x")
    w1 = tf.constant([0.2, 0.1, 0.3, 0.2, 0.4, 0.3], shape=[2, 3], name="w1")
    b1 = tf.constant([-0.3, 0.1, 0.2], shape=[1, 3], name="b1")
    w2 = tf.constant([0.2, 0.5, 0.25], shape=[3, 1], name="w2")
    b2 = tf.constant([-0.3], shape=[1, 1], name="b2")
    a = tf.nn.relu(tf.matmul(x, w1)+b1)
    y = tf.nn.relu(tf.matmul(a, w2)+b2)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    print(sess.run(y))
    
line()
