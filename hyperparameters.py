# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 14:43:29 2022

@author: alexandru.gorunescu
"""

from itertools import product

layers = ['2000,1000,500','1000,500,500','800,400,400','500,500,250']
dropout_1 = ['0.5', '0.6']
dropout_2 = ['0.5', '0.6']
batch_size = ['512']
learning_rate = ['1e-5']
weight_decay = ['1e-4', '1e-3']
gamma = ['1']
reg = ['l1', 'l2']

hyperparams_comb = list(product(layers, dropout_1, dropout_2, learning_rate, batch_size, weight_decay, gamma, reg))

with open('hyperparameters.txt', 'a') as f:
    
    for h in hyperparams_comb:
    
        f.write(';'.join(h) + '\n')


