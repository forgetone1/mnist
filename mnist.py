# 引入tensorflow1.0并禁用v2
from tensorflow_examples_tutorials_mnist import input_data
import mnistutil
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# 导入训练数据和验证数据到mnist
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# x为训练图像,y_为训练图像标签
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# 第一层卷积神经网络配置w_conv1代表的是神经网络中第一层的权重
# b_conv1代表的是神经网络第一层的偏置
W_conv1 = mnistutil.weight_variable([5, 5, 1, 32])
b_conv1 = mnistutil.bias_variable([32])

# 将单张图片从784维向量重新还原为28x28的矩阵图片，-1表示取出所有的数据
x_image = tf.reshape(x, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(mnistutil.conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = mnistutil.max_pool_2x2(h_conv1)

# 第二层卷积层使用64个单元
W_conv2 = mnistutil.weight_variable([5, 5, 32, 64])
b_conv2 = mnistutil.bias_variable([64])
h_conv2 = tf.nn.relu(mnistutil.conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = mnistutil.max_pool_2x2(h_conv2)

# 全连接层
W_fc1 = mnistutil.weight_variable([7 * 7 * 64, 1024])
b_fc1 = mnistutil.bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# 使用Dropout，训练时为0.5，测试时为1，keep_prob表示保留不关闭的神经元的比例
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 把1024维的向量转换成10维，对应10个类别
W_fc2 = mnistutil.weight_variable([1024, 10])
b_fc2 = mnistutil.bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# 交叉熵
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

# 定义train_step
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 验证准确性
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 存储训练的模型
saver = tf.train.Saver()

# 创建Session和变量初始化
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# 标准训练是20000步,这里为节约时间训练1000步
step = 2000
for i in range(step):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:  # 每100步输出一次在验证集上的准确度
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))

    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
print("test accuracy %g" % accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
saver.save(sess, './model/model.ckpt')  # 模型存储的文件夹

writer = tf.summary.FileWriter('./log', tf.get_default_graph())
writer.close()
             