#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 15:27:28 2020

@author: Hook
"""

import numpy as np
from Prospector_2 import prospector2

BUDGET = 1000
cz = 0.1
cy = 1

costs = {0: 0.1, 1:1}

path = 'data/Scaled_HCOF_F2.csv'
df = np.loadtxt(path, skiprows=1, delimiter=',')
X = df[:,:7]
y_true = df[:,7:]

m,d = X.shape

y_experimental = np.full((m, 2), np.nan)
status = np.zeros(m)

top_100 = np.argsort(y_true[:,-1])[-100:]

start_size = 50
initial_exp = np.random.choice(m, size=start_size, replace=False)

for mat in initial_exp:  # status : 0, 2, 4
    status[mat] += 1
    y_experimental[mat][0] = y_true[mat][0]
    status[mat] += 1
    status[mat] += 1  # it looks silly but this makes sense!
    y_experimental[mat][1] = y_true[mat][1]
    status[mat] += 1
    BUDGET -= (costs[0] + costs[1])
    
p = prospector2(X,costs)

while BUDGET >= cy:
    p.fit(y_experimental, status)
    mat, exp = p.pick_next(status)
    print(F'assessing {mat} with experiment {exp}')
    
    BUDGET -= costs[exp]
    status[mat] += 1  # start test
    y_experimental[mat][exp] = y_true[mat][exp]
    status[mat] += 1  # end test
    
    found = sum(1 for i in range(m) if status[i]==4 and i in top_100)
    print(F'Found {found} top 100')
    
    


