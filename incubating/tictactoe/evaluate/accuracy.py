import numpy as np
from sklearn.metrics import accuracy_score

class Accuracy:
	def __init__(self):
		pass
	def eval(self, y_true, y_pred):
		y_pred[y_pred <= 0] = 0
		y_pred = np.floor(y_pred/y_pred.max(axis=1)[:,None])
		return accuracy_score(y_true, y_pred)

	def evalTTT(self, y_true, y_pred):
		y_pred[y_pred <= .5] = 0
		y_pred[y_pred > .5] = 1
		n_equal = np.sum(y_pred==y_true,axis=1)
		print 'n_equal',n_equal
		return len(n_equal[n_equal==len(y_pred[0])])
