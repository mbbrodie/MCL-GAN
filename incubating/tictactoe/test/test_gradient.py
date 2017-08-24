#test_gradient
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


s = ExperimentSettings('test_gradient_settings.txt')
d = Dataset(s)
l = CrossEntropyLoss()
m = MLP(s)
res = Saver(s)
t = BasicTrainer()

t.train(s,m,d,res,l)
print dir(m)
for lay in m.layers:
	print lay.W
	print lay.b 
	print lay.is_output

# train(self, settings, model, data, saver, loss):  