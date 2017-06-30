'''
Simple module for loading CIFAR10 train/test data
Will likely move this to a 'dataset' module
'''
import tensorflow as tf
from keras.datasets import cifar10
from tensorflow.python.framework import dtypes
import numpy as np

'''
Tensorflow currently doesn't play nicely with loading Numpy datasets into queues.
This class stores an index, shuffles the data, and loads the next batch
'''
class Cifar10Data():
    def __init__(self, batch_size=50):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = self.reshape(x_train)
        x_test = self.reshape(x_test)
        y_train = self.to_one_hot(y_train)
        y_test = self.to_one_hot(y_test)
        self.train = (x_train, y_train)
        self.test = (x_test, y_test)

    def to_one_hot(self, labels):
        one_hot = np.zeros((labels.shape[0], 10))
        one_hot[np.arange(labels.shape[0]), labels] = 1
        return one_hot

    def reshape(self, inputs):
        return np.transpose(inputs, (0,2,3,1)) 

    def shuffle_train(self):
        self.train_idx = 0
        shuffle_idx = np.random.shuffle(np.arange(50000))
        self.train[0] = self.train[0][shuffle_idx]
        self.train[1] = self.train[1][shuffle_idx]

    def get_train_batch(self, train_idx, batch_size):
        images, labels = self.train[0][train_idx:train_idx+batch_size], self.train[1][train_idx:train_idx+batch_size]
        return images, labels
    
    def get_test_batch(self):
        self.test_idx = self.test_idx + self.batch_size
        if self.test_idx + self.batch_size > 10000:
            return None
        return self.test[0][self.test_idx:self.test_idx_self.batch_size], self.test[1][self.test_idx:self.test_idx_self.batch_size]

#does NOT return one-hot encoded labels
def get_cifar10_train_data(batch_size=50):
    (x_train, y_train), _ = cifar10.load_data()
    #return tf.contrib.data.Dataset.from_tensor_slices((x_train, y_train))
    
    images, labels = tf.train.batch(
        [x_train, y_train],
        batch_size=batch_size,
        num_threads=-1,
        capacity=2*batch_size)
    return images, labels
    '''
    queue = tf.RandomShuffleQueue(
        capacity=1000,
        min_after_dequeue=10,
        dtypes=[tf.uint8, tf.uint8]
    )
    '''
    #return images, labels
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
