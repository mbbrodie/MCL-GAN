import cPickle as pickle
import numpy as np
import random

class Dataset:
	def __init__(self, settings): 
		self.load_data(settings)
		self.index = 0		

	def load_data(self, settings):
		if '.pkl' in settings.path:
			self.load_pickled_data(settings.path)
			return
		f = open(settings.path, 'r')
		self.X = []
		self.Y = []
		for l in f.readlines():
			x,y = l.split(' ')
			x = map(float,x.split(','))
			y = map(float,y.split(','))
			self.X.append(x)
			self.Y.append(y)
		self.X = np.asarray(self.X)
		self.Y = np.asarray(self.Y)
		self.indexing = range(self.X.shape[0])
		random.shuffle(self.indexing)
		#self.save_data_as_pickle(settings.path) # this explodes file size by ~6x

	def save_data_as_pickle(self, path):
		path = path[:-4]+'.pkl'
		pickle.dump((self.X,self.Y), open(path, 'wb'))

	def load_pickled_data(self, path):
		print 'loading pickled data'
		self.X, self.Y = pickle.load(open(path, 'rb'))
		self.indexing = range(self.X.shape[0])
		random.shuffle(self.indexing)

	def has_batch(self, batch_size):
		batch_size = int(batch_size)
		if self.index + batch_size < len(self.indexing):
			return True
		return False		

	def next_batch(self, batch_size):		
		batch_size = int(batch_size)
		#X = self.X[self.index:self.index+batch_size]
		#Y = self.Y[self.index:self.index+batch_size]
		X = self.X[self.indexing[self.index:self.index+batch_size]]
		Y = self.Y[self.indexing[self.index:self.index+batch_size]]
		self.index = self.index + batch_size
		return X,Y

	def shuffle(self):
		random.shuffle(self.indexing)
		self.index = 0

	def resample(self):
		pos_idx = np.where(self.Y == 1.0)[0]
		pos_X = self.X[pos_idx]
		pos_Y = self.Y[pos_idx]
		n_copies = (self.Y.shape[0] - len(pos_idx)) / len(pos_idx)
		#print 'len pos_idx, n_copies'
		#print len(pos_idx)
		#print n_copies
		resamp_X = np.tile(pos_X, (n_copies,1))
		resamp_Y = np.tile(pos_Y, (n_copies,1))
		self.X = np.concatenate((self.X, resamp_X))
		self.Y = np.concatenate((self.Y, resamp_Y))		
		self.indexing = range(self.X.shape[0])
		random.shuffle(self.indexing)
		#print 'index, Xshape, yshape'
		#print len(self.indexing)
		#print self.X.shape
		#print self.Y.shape
		#import sys; sys.exit()






