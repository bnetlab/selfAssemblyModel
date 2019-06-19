#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 12:18:18 2019

@author: ranap
"""

import numpy as np
import pandas as pd

df1 = pd.read_csv('dataSimulation.csv', index_col=0)

df2 = pd.read_csv('result_phase.csv')

new_df = df1.merge(df2,  how='inner', left_on=['T','tau'], right_on = ['T','tau'])

new_df = new_df.drop( [ "S3"], axis=1)

new_df.columns = ["d", "v", "sigma", "T", "tau", "S2_simulation", "S4_simulation", "S2_analytical", "S4_analytical"]

new_df.to_csv("combined_result.csv",index=None )