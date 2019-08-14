#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 13:46:07 2019

@author: ranap
"""

import numpy as np

f_alpha=True

del_t=0.001
v=1
D=0.5
l=10
m=1
del_G = 1


del_x = 2*D*del_t*np.random.randn()
v_eff = v + del_x/del_t 
del_p = m*del_x/del_t



