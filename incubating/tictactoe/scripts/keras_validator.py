#train validator
import sys
# Add the ptdraft folder path to the sys.path list
sys.path.append('/Users/mike/Documents/ml_lab/phd/proposal/code/MCL-GAN/incubating/tictactoe')
sys.path.append('..')

from settings.run_settings import ExperimentSettings
from model.mlp import MLP
from data.csv_data import Dataset
from results.saver import Saver
from trainer.trainer import BasicTrainer
from loss.loss import *
from evaluate.accuracy import Accuracy


s = ExperimentSettings('validator_settings2.txt')
d = Dataset(s)
d.resample()

X = d.X
Y = d.Y
#keras part
from keras.models import Sequential
from keras.layers import Dense

#create model (could later create based on txt file
model = Sequential()
model.add(Dense(50, input_dim=27, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, nb_epoch=3, batch_size=100)
# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
model.save('keras_validator_model2.h5')

#USE THIS: https://github.com/amir-abdi/keras_to_tensorflow

'''
l = CrossEntropyLoss()
m = MLP(s)
res = Saver(s)
t = BasicTrainer()
e = Accuracy('binary')
t.train(s,m,d,res,l,e)
e.report(d, m, s)
'''
# import gc
# gc.collect()
# y_pred = m.compute(d.X)
# y_pred[y_pred <= .5] = 0
# y_pred[y_pred > .5] = 1
# print y_pred
# y_true = d.Y
# from sklearn.metrics import accuracy_score	
# print accuracy_score(y_true, y_pred)


#print m.compute(d.X)
#print e.evalTTT(d.Y,m.compute(d.X))
