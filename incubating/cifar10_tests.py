import tensorflow as tf
from cifar10_data import *
from cifar10_model import *
'''
class LoadDataTest(tf.test.TestCase):
    def testTrain(self):
        data = Cifar10Data(batch_size=10)
        with self.test_session() as sess:
            images, labels = data.get_train_batch()
            print images.shape
            print labels.shape
            

    def testEval(self):
        with self.test_session():
            pass
            #test = get_cifar10_eval_data()
            #print test
'''

class Cifar10QuickTest(tf.test.TestCase):
    def testModel(self):
        batch_size = 10
        data = Cifar10Data(batch_size=batch_size)
        train_idx = 0 - batch_size
        inputs = tf.placeholder(tf.float32, shape=(10, 32, 32, 3))
        labels = tf.placeholder(tf.float32, shape=(10, 10))
        train_op = train(inputs, labels)
        with self.test_session() as sess:
            train_idx = train_idx + batch_size
            x, y = data.get_train_batch(train_idx, batch_size)
            print x.shape
            print y.shape
            #import sys; sys.exit()
            res = sess.run(train_op, feed_dict={inputs:x.astype(float32), labels: y.astype(float32)})
            print res


if __name__ == '__main__':
    tf.test.main()
            
