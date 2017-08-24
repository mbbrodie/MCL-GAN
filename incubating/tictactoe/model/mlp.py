import numpy as np
import copy
import cPickle as pickle
import sys

class Layer:
    def __init__(self, m, n):
        self.m = m
        self.n = n
        self.is_output = False

    def init_weights_random(self):
        self.W = 0.01*np.random.randn(self.m,self.n)
        self.b = np.ones(self.n) #0.01*np.random.randn(self.n)

    def set_is_output(self, val):
        self.is_output = val

    def compute(self, X):
        if self.is_output:
            self.activation = np.dot(X,self.W) + self.b
            #print self.activation
        else:
            self.activation = np.maximum(.1*(np.dot(X,self.W) + self.b), np.dot(X,self.W) + self.b)
        return self.activation
        

class MLP:
    def __init__(self, settings):
        if hasattr(settings, '.'):
            self.init_layers_from_pretrained(settings.load_weights_path)
        else:        
            self.init_layers(settings.layer_params)        

    def init_layers_from_pretrained(self, path):
        saved = pickle.load( open(path, 'rb') )
        self.layers = saved.layers
        #self.layers = copy.deepcopy(other.layers) 

    def init_layers(self, layer_params):
        #layer param form: [[l1_m, l1_n], [l2_m, l2_n], etc.]
        # the n of the previous layer must match the m of the next layer    
        self.layers = []
        layer_params = eval(layer_params) #string to list of         
        for l in layer_params:            
            layer = Layer(m=l[0], n=l[1])
            layer.init_weights_random()
            self.layers.append(layer)
        #print self.layers       
        self.layers[-1].set_is_output(True)

    def compute(self, X):
        l_input = X
        for l in self.layers:
            l_input = l.compute(l_input) 
        return l_input 
    
    def do_gradient_descent(self, output, X, Y, lr, dout):
        #dout is the output gradient in itially, Thereafter the gradient of each hidden activation
        #dout = (output - Y) / output.shape[0]
        # for W update: TRANSPOSE output of layer before DOT upper level deriv
        # for B: sum upper output deriv by row 
        # for hidden layer deriv: upper deriv DOT upper weight TRANSPOSE
        dW = []
        db = []
        #print 'dout', dout
        for l in range(len(self.layers)-1,-1,-1):
            nxt_dout = np.dot(dout, self.layers[l].W.T)
            # for z in self.layers:
            #     print z.activation            
            #dW.append( np.dot(self.layers[l].activation.T, dout) )
            #db.append( np.sum(dout, axis=0, keepdims=True) )
            #dout = np.dot(dout, self.layers[l].W.T)
            #dout[dout <=0] = 0 #for relu            
            #sys.exit()
            activation = X.T if l == 0 else self.layers[l-1].activation.T
            #dW = np.dot(self.layers[l].activation.T, dout)            
            dW = np.dot(activation, dout)
            # print 'activation',activation
            # print 'dout',dout
            # print 'dw',dW #; sys.exit()            
            db = np.sum(dout, axis=0, keepdims=True)
            #print dW.shape
            #print self.layer[l].W.shape
            self.layers[l].W = self.layers[l].W - lr * dW
            self.layers[l].b = self.layers[l].b - lr * db
            dout = nxt_dout
            #dout[dout <=0] = 0
            #test --> make sure there are no odd loop issues (i.e. dout only takes the last value)            
