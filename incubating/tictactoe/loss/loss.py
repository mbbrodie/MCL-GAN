import numpy as np
class CrossEntropyLoss:
    """cross entropy loss"""
    def __init__(self):
        self.count= 0

    def measure(self, y_in, y_actual):
        """Because we can have 3 possible outputs: x,o,b, we want 3 output vectors predicted x [111000000] o [000010100] and b [000101011]"""
        # print 'measure y_in'
        # print 'yactual, yin'        
        # print yactual
        # print y_in
        #print y_in
        #self.probs = np.exp(y_in) / np.sum(np.exp(y_in), axis=1, keepdims=True) # keep this for other types of predictions
        self.probs = y_in
        self.probs[self.probs <= 0] = 0.000001
        self.log_loss = np.sum(-1.0*(np.log(self.probs)*y_actual)) / y_in.shape[0]        
        #print '-np.log(self.probs*y_actual)',np.sum(-1.0*np.log(self.probs)*y_actual)             
        return self.log_loss
    
    def compute_gradient(self, y_in, y_actual):        
        #print 'in loss.py'
        #print 'y_in', y_in
        #print 'y_act', y_actual  
        # print 'gradient!'  
        self.grad = (y_in - y_actual) / y_in.shape[0]
        self.count = self.count + 1
        # if self.count % 1000 == 0:
        # # if True:
        #       print self.count
        #       print 'y_in', y_in
        #       print 'y_actual', y_actual
        #       print 'out_grad', self.grad
        return self.grad
