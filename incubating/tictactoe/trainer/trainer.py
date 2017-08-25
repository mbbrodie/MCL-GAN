from model import model
import numpy as np


class BasicTrainer:
    '''inits the optimizer, and controls the # epochs, learning rates etc.'''
    def __init__(self):
        pass        

    def train(self, settings, model, data, saver, loss, metric):        
        lr = float(settings.lr)
        for iter in range(0, int(settings.n_iter)):
            if iter % 10000 == 0 and iter > 0:                
                lr = .1 *lr
                print 'lr', lr
            #count = 0
            while data.has_batch(settings.batch_size):                
                #count = count + 1
                #if count > 4:
                #    import sys; sys.exit()
                X,Y = data.next_batch(settings.batch_size)
                output = model.compute(X)
                #print 'Y',Y
                #print 'pred',output
                #print output                
                curr_loss = loss.measure(output, Y)
                # print curr_loss
                #import sys; sys.exit()
                grad = loss.compute_gradient(output,Y)
                model.do_gradient_descent(output, X, Y, lr, grad)                
            metric.report(data, model, settings)
            if saver.is_save_iter():
                saver.save_loss(curr_loss)
                saver.save_model(model)
            saver.increment_iter()
            data.shuffle()            