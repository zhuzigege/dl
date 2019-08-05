# _*_ coding: utf-8 _*_
# __author__ = 'yuezhu'
# create on 2019/7/22

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
datasets = input_data.read_data_sets(train_dir='C:/Users/zhule/IdeaProjects/dl/dataset/mnist', one_hot=True)

def weight_variable(shape):
    init = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init)

def bias_variable(shape):
    init = tf.constant(0.1, shape=shape)
    return tf.Variable(init)

def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])

X_image = tf.reshape(X, [-1,28,28,1])

W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(X_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

h_pool2_flat = tf.reshape(h_pool2,[-1, 7 *7 * 64])
W_fc1 = weight_variable([7 *7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

loss = -tf.reduce_mean(Y * tf.log(y_conv))

train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)

correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(y_conv, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for i in range(20000):
        batch_x, batch_y = datasets.test.next_batch(200)
        sess.run(train_step, feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0})
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.5})
            print("step %d, training accuracy %g"%(i, train_accuracy))
    print(sess.run(accuracy, feed_dict={X: datasets.test.images, Y: datasets.test.labels, keep_prob: 1.0}))
