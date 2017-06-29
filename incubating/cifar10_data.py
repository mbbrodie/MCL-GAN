'''
Simple module for loading CIFAR10 train/test data
Will likely move this to a 'dataset' module
'''
import tensorflow as tf
from keras.datasets import cifar10
from tensorflow.python.framework import dtypes
import numpy 

#does NOT return one-hot encoded labels
def get_cifar10_train_data(batch_size=50):
    (x_train, y_train), _ = cifar10.load_data()
    images, labels = tf.train.batch(
        [x_train, y_train],
        batch_size=batch_size,
        num_threads=1,
        capacity=2*batch_size)
    return images, labels
    #return tf.contrib.data.Dataset.from_tensor_slices((x_train, y_train))

def get_cifar10_eval_data(batch_size=50):
    _, (x_test, y_test) = cifar10.load_data()
    return tf.contrib.data.Dataset.from_tensor_slices((x_test, y_test))

'''
#d = DataSet(x_train, y_train)
d = tf.contrib.data.Dataset.from_tensor_slices((x_train, y_train))
#batches = d.batch(50)
queue = tf.RandomShuffleQueue()
inputs = queue.dequeue_many(50)
#b_iter = batches.make_one_shot_iterator()
'''
'''
print 'b_iter'
print dir(b_iter)
batch = b_iter.get_next()
print 'batch'
print dir(batch)
with tf.Session() as sess:
    for i in range(0,10):
        val = sess.run(inputs)
        print val
'''
