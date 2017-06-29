import tensorflow as tf
slim = tf.contrim.slim

def get_cifar10_quick(inputs):
    with tf.variable_scope(scope, 'vgg_a', [inputs]) as sc:
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            weights_initializer=tf.constate_initializer(0.0),
                            activation=False):
            net = slim.conv2d(inputs, 32, [5, 5], stride=1, padding='SAME',scope='conv1')
            net = slim.pool2d(net, [3, 3], stride=2, scope='pool1')
            net = tf.nn.relu(net)
            net = slim.conv2d(net, 32, [5, 5], stride=1, padding='SAME',scope='conv2')
            net = tf.nn.relu(net)
            net = slim.pool2d(net, [3, 3], stride=2, scope='pool2') 
            net = slim.conv2d(net, 64, [5, 5], stride=1, padding='SAME',scope='conv3')
            net = tf.nn.relu(net)
            net = slim.pool2d(net, [3, 3], stride=2, scope='pool3') 
            net = slim.stack(net, slim.fully_connected, [64, 10], scope='fc')
    return net
