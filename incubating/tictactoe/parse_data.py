'''
x_idx
o_idx
count # of blanks
while len(blanks) < 5 (you could do 3 as well):
    if len(x_idx) > len(o_idx)
        randomly select idx


it might be easier to just program solutions.
There are 8 ways for both players to win.
5 moves - X wins

6 moves - Y wins

7 moves - X wins

8 moves - Y wins

import copy
play_index = range(9)
For n iter:
    play_index = random.shuffle(play_index)
    a = 9 array of 'b'
    is_x_turn = True
    starting_boards = []
    playing = True
    while playing 
        if n_b < 5:
            if is_winner(a,'x') or is_winner(a,'o'):
                #write starting boards and solutions
        #play
        piece
        is_x_turn = not is_x_turn

f = open('data.csv', 'r')
out = open('tictactoe.csv','w')
for l in f.readlines():
    arr = l.split(',')
    n_blank = len([i for i in arr if i == 'b'])
    if n_blank > 4:
        out.write(l)
out.close()
'''
import random
import itertools
import copy


def get_winner(board):
    x_idx = set([idx for idx,val in enumerate(board) if val == 'x'])
    o_idx = set([idx for idx,val in enumerate(board) if val == 'o'])
    winning_sets = [set([0,1,2]), set([3,4,5]), set([6,7,8]), set([0,3,6]), set([1,4,7]), set([2,5,8]), set([0,4,8]), set([2,4,6])]
    for s in winning_sets:
        if s.issubset(x_idx):
            return 'won'
        if s.issubset(o_idx):
            return 'lost'
    n_b = 9 - len(x_idx) - len(o_idx)
    if n_b == 0:
        return 'draw'
    return None


out = open('tictactoe.csv','w')
a = range(9)
play_orders = itertools.permutations(a,9)
n_iter = 50000
play_indexes = []
#for p in play_orders:
#    play_indexes.append(p)
#random.shuffle(play_indexes)
#for n in range(0,n_iter):
for p in play_orders:
    play_order = p #play_indexes[n]
    curr_board = ['b' for i in range(9)]
    starting_boards = []
    for idx,val in enumerate(play_order):
        if idx % 2 == 0: # x turn
            curr_board[val] = 'x' 
        else:
            curr_board[val] = 'o' 
        if idx > 0 and idx < 4:
            starting_boards.append(copy.deepcopy(curr_board)) # need a deep copy?

        if idx > 3:
            winner = get_winner(curr_board) # returns 'x' 'o' 'draw' or None (game not finished yet)
            if winner is not None:
                #write starting boards and end board
                final_board = ','.join(curr_board)
                for sb in starting_boards:
                    out.write(','.join(sb))
                    out.write(','+winner+',')
                    out.write(final_board + '\n')
                break
        
out.close()

        
            













