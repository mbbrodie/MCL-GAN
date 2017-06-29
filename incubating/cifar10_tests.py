import tensorflow as tf
from cifar10_data import *

class LoadDataTest(tf.test.TestCase):
    def testTrain(self):
        with self.test_session():
            images, labels = get_cifar10_train_data()
            print images.eval()

    def testEval(self):
        with self.test_session():
            test = get_cifar10_eval_data()


if __name__ == '__main__':
    tf.test.main()
            
