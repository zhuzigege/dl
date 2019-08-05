# _*_ coding: utf-8 _*_
# __author__ = 'yuezhu'
# create on 2019/7/22

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
datasets = input_data.read_data_sets(train_dir='C:/Users/zhule/IdeaProjects/dl/dataset/mnist', one_hot=True)

X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])

W = tf.Variable(tf.random_uniform([784, 10], -1.0, 1.0))
b = tf.Variable(tf.zeros(10))

y_ = tf.nn.softmax(tf.matmul(X, W) + b)

loss = -tf.reduce_mean(Y * tf.log(y_))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    for i in range(1000):
        batch_x, batch_y = datasets.train.next_batch(500)
        sess.run(train_step, feed_dict={X: batch_x, Y: batch_y})

    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print(sess.run(accuracy, feed_dict={X: datasets.test.images, Y: datasets.test.labels}))
