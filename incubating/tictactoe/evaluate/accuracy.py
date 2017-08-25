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
		y_pred = np.floor(y_pred/y_pred.max(axis=1)[:,None])
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

	def report(self, y_true, y_pred):
		print self.name
		print self.eval_f(y_true, y_pred)