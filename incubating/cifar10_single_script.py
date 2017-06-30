import tensorflow as tf
import numpy as np
from cifar10_data import *
slim = tf.contrib.slim

#CIFAR10 Quick Model
#making this in a single file---since passing tf.placeholders by function
#seems to produce errors
inputs = tf.placeholder(tf.float32, shape=(10, 32, 32, 3))
labels = tf.placeholder(tf.float32, shape=(10, 10))
with slim.arg_scope([slim.conv2d, slim.fully_connected],
                    weights_initializer=tf.constant_initializer(0.0),
                    activation_fn=None):
    net = slim.conv2d(inputs, 32, [5, 5], stride=1, padding='SAME',scope='conv1')
    net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
    net = tf.nn.relu(net)
    net = slim.conv2d(net, 32, [5, 5], stride=1, padding='SAME',scope='conv2')
    net = tf.nn.relu(net)
    net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool2') 
    net = slim.conv2d(net, 64, [5, 5], stride=1, padding='VALID',scope='conv3')
    net = tf.nn.relu(net)
    net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool3') 
    net = slim.stack(net, slim.fully_connected, [64, 10], scope='fc')
    predictions = tf.squeeze(net,[1,2])

loss = slim.losses.softmax_cross_entropy(predictions, labels)
optimizer = tf.train.GradientDescentOptimizer(0.1)
train_op = slim.learning.create_train_op(loss, optimizer)
logdir = 'cifar10_logs'

#final_loss = slim.learning.train(number_of_steps=10)
    #save_summaries_secs=300,
    #save_interval_secs=600)

def main():
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        batch_size = 10
        data = Cifar10Data(batch_size=batch_size)
        train_idx = 0 - batch_size
        train_idx = train_idx + batch_size
        x, y = data.get_train_batch(train_idx, batch_size)
        res = sess.run(train_op, feed_dict={inputs:x.astype(np.float32), labels: y.astype(np.float32)})
        print res

if __name__ == '__main__':
    main()
