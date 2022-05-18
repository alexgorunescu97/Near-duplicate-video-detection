# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 13:06:11 2022

@author: alexandru.gorunescu
"""

import os

max_acc = float('-inf')
res = []
for root, dirs, files in os.walk('results/results_vgg'):
    n = root.split('_')
    if 'dev' in root:
        with open(os.path.join(root, files[-1]), 'r') as f:
            lines = f.readlines()
            curr_acc = float(lines[0][20: -1].split('%')[0])
            res.append([curr_acc, n[-2]])
            if curr_acc > max_acc:
                max_acc = curr_acc
                conf_n = root
sorted_res = sorted(res, key=lambda x : x[0])
print(sorted_res)
print(f'Max acc {max_acc} at conf {conf_n}')
            