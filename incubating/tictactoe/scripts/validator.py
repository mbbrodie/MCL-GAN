#train validator
import sys
# Add the ptdraft folder path to the sys.path list
sys.path.append('/Users/mike/Documents/ml_lab/phd/proposal/code/MCL-GAN/incubating/tictactoe')

from settings.run_settings import ExperimentSettings
from model.mlp import MLP
from data.csv_data import Dataset
from results.saver import Saver
from trainer.trainer import BasicTrainer
from loss.loss import *
from evaluate.accuracy import Accuracy


s = ExperimentSettings('validator_settings.txt')
d = Dataset(s)
d.resample()
l = CrossEntropyLoss()
m = MLP(s)
res = Saver(s)
t = BasicTrainer()
e = Accuracy('binary')
t.train(s,m,d,res,l,e)
e.report(d, m, s)
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