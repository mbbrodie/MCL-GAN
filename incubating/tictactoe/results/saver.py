import os
import cPickle as pickle


class Saver:
    def __init__(self, settings):
        self.curr_iter = 0
        self.loss_save_dir = settings.loss_save_dir
        if not os.path.exists(self.loss_save_dir):
            os.makedirs(self.loss_save_dir)
        f = open(self.loss_save_dir + 'loss.txt','w')
        f.write(settings.to_string() + '\n')
        f.close()
        self.model_save_dir = settings.model_save_dir
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        self.save_after_n_iter = int(settings.save_after_n_iter)

    def increment_iter(self):
        self.curr_iter = self.curr_iter + 1

    def is_save_iter(self):
        return self.curr_iter % self.save_after_n_iter == 0
            
    def save_model(self, model):
        pickle.dump(model, open(self.model_save_dir+'iter'+str(self.curr_iter)+'.pkl', 'wb'))

    def save_loss(self, loss):
        f = open(self.loss_save_dir + 'loss.txt','a')
        f.write('iter '+str(self.curr_iter)+': '+str(loss)+'\n')
        f.close()

'''
record_loss

'''

