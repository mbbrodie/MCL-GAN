import random
import numpy as np
import itertools
import copy

def convert_to_onehot(arr, is_x=True):
    encoded = np.zeros(30) if is_x else np.zeros(27)
    for idx,val in enumerate(arr):
        if val == 'x':
            encoded[idx] = 1.0
        elif val == 'o':
            encoded[9+idx] = 1.0
        elif val == 'b':
            encoded[18+idx] = 1.0
        elif val == 'won':
            encoded[27] = 1.0
        elif val == 'lost':
            encoded[28] = 1.0
        else:
            encoded[29] = 1.0
    return encoded

f = open('tictactoe_standardized.csv','r')
out = open('tictactoe_standardized_processed.csv','w')
for l in f.readlines():
    l = l[:-1].split(',')
    x = l[:10]
    y = l[10:]
    x_n = [str(i) for i in convert_to_onehot(x)]
    y_n = [str(i) for i in convert_to_onehot(y, is_x=False)]
    out.write(','.join(x_n) +' '+','.join(y_n)+'\n')

out.close()
            
