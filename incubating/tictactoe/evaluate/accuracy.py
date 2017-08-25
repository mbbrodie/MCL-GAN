import numpy as np
from sklearn.metrics import accuracy_score

class Accuracy:
	def __init__(self, o_type):	
		self.name = 'accuracy: '	
		if o_type == 'binary':
			self.eval_f = self.eval_binary
		elif o_type == 'multiclass':
			self.eval_f = self.eval_multiclass
		elif o_type == 'ttt': #multilabel?
			self.eval_f = self.eval_TTT


	def eval_multiclass(self, y_true, y_pred):
		y_pred[y_pred <= 0] = 0		
		if np.count_nonzero(y_pred) > 1:
			y_pred = np.floor(y_pred/y_pred.max(axis=1)[:,None])
		else:	
			y_pred[:] = 0		
		return accuracy_score(y_true, y_pred)

	def eval_TTT(self, y_true, y_pred):
		y_pred[y_pred <= .5] = 0
		y_pred[y_pred > .5] = 1
		n_equal = np.sum(y_pred==y_true,axis=1)
		print 'n_equal',n_equal
		return len(n_equal[n_equal==len(y_pred[0])])

	def eval_binary(self, y_true, y_pred):
		y_pred[y_pred <= .5] = 0
		y_pred[y_pred > .5] = 1
		return accuracy_score(y_true, y_pred)

	def report(self, data, model, settings):
		data.shuffle() 
		metric = 0
		n_batches = 0
		while data.has_batch(settings.batch_size): 
			n_batches = n_batches + 1
			X,y_true = data.next_batch(settings.batch_size)
			y_pred = model.compute(X)
			metric = metric + self.eval_f(y_true, y_pred)
		print self.name
		print 1.0 * metric / n_batches


		