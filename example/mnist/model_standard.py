# _*_ coding: utf-8 _*_
# __author__ = 'yuezhu'
# create on 2019/7/24

import os.path
import time
import math

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import mnist
from tensorflow.examples.tutorials.mnist import input_data

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 20000, 'Number of steps to run trainer.')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('batch_size', 100, 'Batch size.  '
                                        'Must divide evenly into the dataset sizes.')
flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                                         'for unit testing.')


class CNNet():
    IMAGE_PIXELS = 784

    def inference(images_placeholder, hidden1_num, hidden2_num):
        with tf.name_scope('hidden1'):
            W_fc1 = tf.Variable(tf.truncated_normal([CNNet.IMAGE_PIXELS, hidden1_num], stddev=1.0 / math.sqrt(float(CNNet.IMAGE_PIXELS))), name='weights')
            b_fc1 = tf.Variable(tf.zeros([hidden1_num]), name='biases')
            h_fc1 = tf.nn.relu(tf.matmul(images_placeholder, W_fc1) + b_fc1)
        with tf.name_scope('hidden2'):
            W_fc2 = tf.Variable(tf.truncated_normal([hidden1_num, hidden2_num], stddev=1.0 / math.sqrt(float(hidden1_num))), name='weights')
            b_fc2 = tf.Variable(tf.zeros([hidden2_num]), name='biases')
            h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
        with tf.name_scope('hidden3'):
            W_fc3 = tf.Variable(tf.truncated_normal([hidden2_num, 10], stddev=1.0 / math.sqrt(float(hidden2_num))), name='weights')
            b_fc3 = tf.Variable(tf.zeros([10]), name='biases')
            logits = tf.matmul(h_fc2, W_fc3) + b_fc3
        return logits

    def loss(logits, labels_placeholder):
        labels_placeholder = tf.to_int64(labels_placeholder)
        return tf.losses.sparse_softmax_cross_entropy(labels=labels_placeholder, logits=logits)

    def train(loss, learn_rate):
        tf.summary.scalar('loss', loss)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        return tf.train.GradientDescentOptimizer(learning_rate=learn_rate).minimize(loss, global_step=global_step)

    def evaluation(logits, labels_placeholder):
        correct = tf.nn.in_top_k(logits, labels_placeholder, 1)
        return tf.reduce_sum(tf.cast(correct, tf.int32))

def placeholder_inputs(batch_size):
    images_placeholder = tf.placeholder(tf.float32, shape=[batch_size, CNNet.IMAGE_PIXELS])
    labels_placeholder = tf.placeholder(tf.int32, shape=[batch_size])
    return images_placeholder, labels_placeholder

def do_eval(sess, eval_correct, images_placeholder, labels_placeholder, dataset):
    true_count = 0  # Counts the number of correct predictions.
    steps_per_epoch = dataset.num_examples // FLAGS.batch_size
    num_examples = steps_per_epoch * FLAGS.batch_size
    for step in xrange(steps_per_epoch):
        batch_x, batch_y = dataset.next_batch(FLAGS.batch_size)
        true_count += sess.run(eval_correct, feed_dict={images_placeholder:batch_x, labels_placeholder:batch_y})
    precision = true_count / num_examples
    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
          (num_examples, true_count, precision))

def run_training():
    print(FLAGS.batch_size)
    images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size)
    datasets = input_data.read_data_sets(train_dir='C:/Users/zhule/IdeaProjects/dl/dataset/mnist', one_hot=False)

    inference = CNNet.inference(images_placeholder, FLAGS.hidden1, FLAGS.hidden2)
    loss = CNNet.loss(inference, labels_placeholder)
    training = CNNet.train(loss, FLAGS.learning_rate)
    eval_correct = CNNet.evaluation(inference, labels_placeholder)


    # Build the summary Tensor based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    # Add the variable initializer Op.
    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
        sess.run(init)
        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            batch_x, batch_y = datasets.train.next_batch(FLAGS.batch_size)

            _, loss_value = sess.run([training, loss],
                                     feed_dict ={images_placeholder:batch_x, labels_placeholder: batch_y})

            duration = time.time() - start_time

            if step % 100 == 0:
                correct = sess.run([eval_correct],
                                   feed_dict ={images_placeholder:batch_x, labels_placeholder: batch_y})
                print('Step %d: loss = %.2f correct = %.2f  (%.3f sec)' % (step, loss_value, correct[0], duration))
                summary_str = sess.run(summary_op, feed_dict ={images_placeholder:batch_x, labels_placeholder: batch_y})
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            if (step + 1) % 1000 == 0:
                checkpoint_file = os.path.join(FLAGS.train_dir, 'checkpoint')
                saver.save(sess, checkpoint_file, global_step=step)

                # Evaluate against the training set.
                print('Training Data Eval:')
                do_eval(sess, eval_correct, images_placeholder, labels_placeholder, datasets.train)

                print('Training Data Eval:')
                do_eval(sess, eval_correct, images_placeholder, labels_placeholder, datasets.validation)

                print('Training Data Eval:')
                do_eval(sess, eval_correct, images_placeholder, labels_placeholder, datasets.test)

run_training()