#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 17:07:58 2020

@author: ranap
"""

# simple fit

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


x = np.random.uniform(0., 100., 100)
y = 3. * x + 2. + np.random.normal(0., 10., 100)
plt.plot(x, y, '.')



def line(x, a, b, c):
    return c * (a * x + b)


popt, pcov = curve_fit(line, x, y)

