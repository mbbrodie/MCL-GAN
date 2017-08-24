from itertools import combinations
import numpy as np


def n_wins(board):
    n = 0
    indexes = set([idx for idx,val in enumerate(board) if val == 1])
    winning_sets = [set([0,1,2]), set([3,4,5]), set([6,7,8]), set([0,3,6]), set([1,4,7]), set([2,5,8]), set([0,4,8]), set([2,4,6])]
    for s in winning_sets:
        if s.issubset(indexes):
            n = n + 1 
    return n

def is_overlap(board):
    x_idx = set([idx for idx,val in enumerate(board[0:9]) if val == 1]) 
    o_idx = set([idx for idx,val in enumerate(board[9:18]) if val == 1]) 
    b_idx = set([idx for idx,val in enumerate(board[18:]) if val == 1]) 
    #print board
    #print x_idx
    #print o_idx
    #print b_idx
    if x_idx.isdisjoint(o_idx) and x_idx.isdisjoint(b_idx) and o_idx.isdisjoint(b_idx):
        return False
    return True

   
def is_valid(board):
    if is_overlap(board):
        #print 'a'
        return 0.0
    n_b = np.count_nonzero(board[18:])
    if n_b > 4:
        #print 'b'
        return 0.0
    n_x = np.count_nonzero(board[0:9])
    n_o = np.count_nonzero(board[9:18])
    if abs(n_x - n_o) > 1:
        #print 'c'
        return 0.0
    if n_x + n_o + n_b <> 9:
        #print 'not 9 moves'
        return 0.0
    n_wins_x = n_wins(board[0:9])
    n_wins_o = n_wins(board[9:18])
    if n_wins_x + n_wins_o > 1:
        #print 'd'
        return 0.0
    if n_wins_x + n_wins_o == 0 and n_b > 0:
        #print 'e'
        return 0.0
    return 1.0

'''
import numpy as np
x_wins = np.zeros(27)
x_wins[9:12] = 1
x_wins[5:7] = 1

print x_wins
print n_wins(x_wins[9:18])
print x_wins
print is_overlap(x_wins)
print is_valid(x_wins)
import sys; sys.exit()
'''

#generate 30 choose 9 possible array combinations
#first try out with 5 choose 2
n = 27
k = 9
comb = combinations(range(n), k)
out = open('validator_train.csv','w')
for c in comb:
    arr = np.zeros(27)
    arr[list(c)] = 1.0
    v = is_valid(arr)
    arr = [str(i) for i in arr]
    #out.write(str(c)[1:-1].replace(" ", "") +' '+str(v)+'\n')
    out.write(','.join(arr) +' '+str(v)+'\n')
out.close()
    
    
