#test script
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


s = ExperimentSettings('test_settings_ttt.txt')
d = Dataset(s)
l = CrossEntropyLoss()
m = MLP(s)
res = Saver(s)
t = BasicTrainer()

t.train(s,m,d,res,l)
e = Accuracy()
print m.compute(d.X)
print e.evalTTT(d.Y,m.compute(d.X))

# train(self, settings, model, data, saver, loss):  

