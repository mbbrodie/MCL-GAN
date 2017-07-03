from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import argparse
from cifar10_data import *
slim = tf.contrib.slim

def cifar10quick(inputs):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                    weights_initializer=tf.random_normal_initializer(stddev=0.01),
                    #weights_initializer=tf.constant_initializer(0.01),
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
    return predictions

def discriminator(inputs):
    with tf.variable_scope('d1'):
        net = tf.layers.dense(inputs, 20)
    net = tf.nn.relu(net)
    with tf.variable_scope('d2'):
        net = tf.layers.dense(net, 20)
    net = tf.nn.relu(net)
    with tf.variable_scope('d3'):
        net = tf.layers.dense(net, 2) # change to fit number of layers
    return net

def optimizer(loss, var_list, initial_learning_rate):
    decay = 0.95
    #num_decay_steps = 150
    num_decay_steps = 1000
    batch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        initial_learning_rate,
        batch,
        num_decay_steps,
        decay,
        staircase=True
    )
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
        loss,
        global_step=batch,
        var_list=var_list
    )
    return optimizer

#define the MODEL class
class EnsDis(object):
    def __init__(self, lr, batch_size):
        self.lr = lr
        self.batch_size = batch_size
        self.create_ensemble()
        
    def create_ensemble(self):
        self.inputs = tf.placeholder(tf.float32, shape=(self.batch_size, 32, 32, 3))
        self.labels = tf.placeholder(tf.float32, shape=(self.batch_size, 10))
        with tf.variable_scope('m1'):
            self.m1 = cifar10quick(self.inputs)
        with tf.variable_scope('m2'):
            self.m2 = cifar10quick(self.inputs)
        #add more ensemble members here...

        #normal losses
        self.loss_m1 = tf.nn.softmax_cross_entropy_with_logits(logits=self.m1, labels=self.labels)
        self.loss_m2 = tf.nn.softmax_cross_entropy_with_logits(logits=self.m2, labels=self.labels)

        #discriminator
        with tf.variable_scope('disc') as d_scope:
            self.D1 = discriminator(self.m1)        
            d_scope.reuse_variables()
            self.D2 = discriminator(self.m2)

        self.m1_labels = tf.placeholder(tf.float32, shape=(self.batch_size, 2))
        self.m2_labels = tf.placeholder(tf.float32, shape=(self.batch_size, 2))
        #uniqueness loss
        loss_unique_m1 = tf.nn.softmax_cross_entropy_with_logits(logits=self.D1, labels=self.m1_labels)
        loss_unique_m2 = tf.nn.softmax_cross_entropy_with_logits(logits=self.D2, labels=self.m2_labels)
        self.total_unique_loss = loss_unique_m1 + loss_unique_m2

        #optimizers
        self.m1_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='m1')
        self.m2_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='m2')
        self.d_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='disc')
        self.opt_m2 = optimizer(self.loss_m2, self.m2_params, self.lr)
        self.opt_d = optimizer(self.total_unique_loss, self.d_params, self.lr)
        self.opt_m1 = optimizer(self.loss_m1, self.m1_params, self.lr)
        

    def train(self):
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            #create ensemble
            #load cifar_data
            data = Cifar10Data(batch_size=self.batch_size)
            m1_labels = np.tile(np.asarray([1,0]), (self.batch_size,1)) #switch out 10 for self.batch_size once it's working
            m2_labels = np.tile(np.asarray([0,1]), (self.batch_size,1))
            train_idx = 0
            for step in xrange(5000):
                #load batch
                batch_inputs, batch_labels = data.get_train_batch(train_idx, self.batch_size) 
                #train ensemble models separately
                batch_loss_m1, _ = sess.run([self.loss_m1, self.opt_m1], {self.inputs: batch_inputs, self.labels: batch_labels})
                batch_loss_m2, _ = sess.run([self.loss_m2, self.opt_m2], {self.inputs: batch_inputs, self.labels: batch_labels})
                if step > 600:
                    #train discriminator
                    batch_unique_loss, _ = sess.run([self.total_unique_loss, self.opt_d], {
                        self.inputs: batch_inputs,
                        self.labels: batch_labels,
                        self.m1_labels: m1_labels,
                        self.m2_labels: m2_labels
                    })
        
                #print losses
                if step % 100 == 0:
                    print('loss_m1', batch_loss_m1)
                    print('loss_m2', batch_loss_m2)
                    if step > 600:
                        print('loss_unique', batch_unique_loss)
            
                train_idx = train_idx + 10
                
                
            
                        
def main(args):
    ensemble = EnsDis(
        args.learning_rate,
        args.batch_size
    )
    ensemble.train()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning-rate', type=float, default=0.0001,
                        help='the learning rate for optimization weight updates')
    parser.add_argument('--batch-size', type=int, default=50,
                        help='the size of a single training batch')

    return parser.parse_args()

if __name__ == '__main__':
    main(parse_args())
