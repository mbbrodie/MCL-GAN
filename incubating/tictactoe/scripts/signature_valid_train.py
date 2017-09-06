import cPickle as pickle
import tensorflow as tf
import numpy as np
import sys
from keras.models import load_model
import tflearn

from keras import backend as K
#K.set_floatx('float64')
# Add the ptdraft folder path to the sys.path list
sys.path.append('..')

from settings.run_settings import ExperimentSettings
from data.csv_data import Dataset
from model.mlp import MLP


def prelu(logit):
    alpha = 0.1#0.001 * logit
    return tf.maximum(logit, tf.multiply(alpha,logit))
    
    

class Pretrained:
    def __init__(self, path):
        self.model = pickle.load( open(path, 'rb'))
        
    def convert_ttt_to_tf(self, m_id):
        #layer_params:[(30,100),(100,200),(200,27)]    
        self.W = {
            #'l0' : tf.Variable(self.model.layers[0].W.astype(np.float64), name=str(m_id)+"_W0", dtype=tf.float64),
            'l0' : tf.Variable(np.asarray(self.model.layers[0].W, dtype='float32'), name=str(m_id)+"_W0"),
            'l1' : tf.Variable(np.asarray(self.model.layers[1].W, dtype='float32'), name=str(m_id)+"_W1"),
            'l2' : tf.Variable(np.asarray(self.model.layers[2].W, dtype='float32'), name=str(m_id)+"_W2")
        }
        self.b = {
            #'b0' : tf.Variable(self.model.layers[0].b.astype(np.float64), name=str(m_id)+"b0", dtype=tf.float64),
            'l0' : tf.Variable(np.asarray(self.model.layers[0].b, dtype='float32'), name=str(m_id)+"_b0"),
            'l1' : tf.Variable(np.asarray(self.model.layers[1].b, dtype='float32'), name=str(m_id)+"_b1"),
            'l2' : tf.Variable(np.asarray(self.model.layers[2].b, dtype='float32'), name=str(m_id)+"_b2")
        } 

    def construct_pred(self, _X):
        #l0 = tf.nn.relu(tf.add(tf.matmul(_X,tf.cast(self.W['l0'], tf.float64)), tf.cast(self.b['l0']), tf.float64)) # RELU vs PRELU (original)
        #l0 = tf.nn.relu(tf.add(tf.matmul(_X,self.W['l0']), self.b['l0'])) # RELU vs PRELU (original)
        #l1 = tf.nn.relu(tf.add(tf.matmul(l0,self.W['l1']), self.b['l1']))
        l0 = prelu(tf.add(tf.matmul(_X,self.W['l0']), self.b['l0'])) # RELU vs PRELU (original)
        l1 = prelu(tf.add(tf.matmul(l0,self.W['l1']), self.b['l1']))
        #return tf.add(tf.matmul(l1,self.W['l2']), self.b['l2'])
        return tf.nn.relu(tf.add(tf.matmul(l1,self.W['l2']), self.b['l2']))

class Signature:
    def __init__(self,n_models):
        n_input,n_hidden,n_out = 27,150,n_models
        self.W = {
            'l0' : tf.Variable(tf.truncated_normal([n_input, n_hidden], stddev=0.01), name="s_W0"),
            'l1' : tf.Variable(tf.truncated_normal([n_hidden, n_out], stddev=0.01), name="s_W1")
            }
        self.b = {
            'l0' : tf.Variable(tf.truncated_normal([n_hidden], stddev=0.01), name="s_b0"),
            'l1' : tf.Variable(tf.truncated_normal([n_out], stddev=0.01), name="s_b1")
        }

    def construct_pred(self, _X):
        #l0 = tf.nn.relu(tf.add(tf.matmul(_X,self.W['l0']), self.b['l0']))
        l0 = prelu(tf.add(tf.matmul(_X,self.W['l0']), self.b['l0']))
        return tf.add(tf.matmul(l0,self.W['l1']), self.b['l1'])

#load keras_validator_model2.h5
class Validator:
    def __init__(self, path):
        self.model = load_model(path)
        print(type(self.model.layers[0].get_weights()[0]))
        self.convert_keras_to_tf()

    def convert_keras_to_tf(self,m_id='valid'):
        #layer_params:[(30,100),(100,200),(200,27)]    
        self.W = {
            'l0' : tf.Variable(self.model.layers[0].get_weights()[0], name=m_id+"_W0", trainable=False),
            'l1' : tf.Variable(self.model.layers[1].get_weights()[0], name=m_id+"_W1", trainable=False),
            'l2' : tf.Variable(self.model.layers[2].get_weights()[0], name=m_id+"_W2", trainable=False)
        }
        self.b = {
            'l0' : tf.Variable(self.model.layers[0].get_weights()[1], name=str(m_id)+"b0", trainable=False),
            'l1' : tf.Variable(self.model.layers[1].get_weights()[1], name=str(m_id)+"b1", trainable=False),
            'l2' : tf.Variable(self.model.layers[2].get_weights()[1], name=str(m_id)+"b2", trainable=False)
        }

    def construct_pred(self, _X):
        l0 = tf.nn.relu(tf.add(tf.matmul(_X,self.W['l0']), self.b['l0'])) # RELU vs PRELU (original)
        l1 = tf.nn.relu(tf.add(tf.matmul(l0,self.W['l1']), self.b['l1']))
        #l0 = tflearn.prelu(tf.add(tf.matmul(_X,self.W['l0']), self.b['l0'])) # RELU vs PRELU (original)
        #l1 = tflearn.prelu(tf.add(tf.matmul(l0,self.W['l1']), self.b['l1']))
        return tf.clip_by_value(tf.sigmoid(tf.add(tf.matmul(l1,self.W['l2']), self.b['l2'])),.0000001,.9999999)



s = ExperimentSettings('signature_settings.txt')
#set num models in ensemble (change this to a load script)
ens_size = int(s.n_models)
batch_size = int(s.batch_size)
lr = float(s.lr)
pretrained_path = s.pretrained_path
validator_path = s.validator_path
n_iter = int(s.n_iter)

#placeholders for X and Y batches
X = tf.placeholder(tf.float32, [batch_size, 30])
#Y = tf.placeholder("float", [n])

#construct pretrained
ens = []
for i in range(ens_size):
    ens.append(Pretrained(pretrained_path))
for i in range(ens_size):
    ens[i].convert_ttt_to_tf(i)


#load validator model
valid = Validator(validator_path)

#Signature and targets
#create target 'y' values (one hot) for each model based on batch size
signature = Signature(ens_size)

ens_preds = []
sig_preds = []
valid_preds = []
sig_targ = []

for i in range(ens_size):
    pred = ens[i].construct_pred(X) 
    ens_preds.append(pred)
    valid_preds.append(valid.construct_pred(pred)) # will this work?
    sig_preds.append(signature.construct_pred(pred))
    st = np.zeros((batch_size, ens_size))
    st[:,i] = 1.0
    st = tf.constant(st)
    sig_targ.append(st)


sig_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=sig_preds[0], labels=sig_targ[0])) + \
                            0#tf.reduce_mean(-tf.reduce_sum(tf.log(valid_preds[0])))
for i in range(1,ens_size):
    sig_cost = sig_cost +  \
                tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=sig_preds[i], labels=sig_targ[i])) + \
                0#tf.reduce_mean(-tf.reduce_sum(tf.log(valid_preds[i])))

print valid_preds[0]
print 'just printed valid preds 0'
valid_cost = tf.reduce_mean(-tf.reduce_sum(tf.log(valid_preds[0])))
for i in range (1,ens_size):
    valid_cost = valid_cost + tf.reduce_sum(-tf.reduce_sum(tf.log(valid_preds[i])))

#add together different costs
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prd,y)) # example
#first_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"scope/prefix/for/first/vars")
#first_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
sig_params_to_train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
valid_params_to_train= [v for v in sig_params_to_train if 's_' not in v.name]
#print first_train_vars
#print type(first_train_vars)
#for i in first_train_vars:
#    print i.name

#valid_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='valid')
#params_to_train = tf.convert_to_tensor(list(set(first_train_vars) - set(valid_vars)))
#print params_to_train


#params_to_train
sig_opt = tf.train.AdamOptimizer(learning_rate = lr).minimize(sig_cost, var_list=sig_params_to_train)
valid_opt = tf.train.AdamOptimizer(learning_rate = lr).minimize(valid_cost, var_list=valid_params_to_train)

###YOU can use the 'params_to_train' parameter to train all non-validator weights simultaneously, or on different schedules
# e.g. train signature, then lock signature weights to push more changes to the models themselves.

sess = tf.Session()
from tensorflow.python import debug as tf_debug
#sess = tf_debug.LocalCLIDebugWrapperSession(sess)
#sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
init_op = tf.global_variables_initializer()
sess.run(init_op)
d = Dataset(s)
d.shuffle()
for step in xrange(n_iter):
    count = 0
    sig_batch_loss,valid_batch_loss = 0,0
    is_sig_train = False
    while d.has_batch(batch_size):
        count = count + 1
        X_,Y_ = d.next_batch(batch_size) 
        #train ensemble models separately
        if count % 1000 == 0:
            is_sig_train = not is_sig_train
        if is_sig_train:
            _, sig_batch_loss = sess.run([sig_opt, sig_cost], {X: X_})
            #_, sig_batch_loss, s_preds = sess.run([sig_opt, sig_cost, sig_preds], {X: X_})
        else:
            _, valid_batch_loss = sess.run([valid_opt, valid_cost], {X: X_})
            #_, valid_batch_loss, s_preds = sess.run([valid_opt, valid_cost, sig_preds], {X: X_})
        if count % 1000 == 0: 
            print 'sig_loss,valid_loss'
            print sig_batch_loss,valid_batch_loss


    





