# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 14:43:29 2022

@author: alexandru.gorunescu
"""

from itertools import product

layers = ['2000,1000,1000']
dropout_1 = ['1']
dropout_2 = ['1']
batch_size = ['2048']
learning_rate = ['1e-5','1e-4','1e-3']
weight_decay = ['1e-5']
gamma = ['1']
reg = ['l1', 'l2']

hyperparams_comb = list(product(layers, dropout_1, dropout_2, learning_rate, batch_size, weight_decay, gamma, reg))

with open('hyperparameters.txt', 'w') as f:
    
    for h in hyperparams_comb:
    
        f.write(';'.join(h) + '\n')


