#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 13:02:14 2019

@author: ranap

V = l/mu
D = 2 * sigma^2 = (2*l^2) / lambda 

Args:
    N: number of particle
Return:
    dataframe with probabilty 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from scipy.stats import invgauss
import math
import time
import sys
from joblib import Parallel, delayed 

def brownian(d,v,sigma,t_bin=0.001):
    
    # simulate brownian motion  
    pos= 0
    t= 1
    v_t = v*t_bin
    while pos < d:
        pos = pos + v_t + (sigma * np.random.randn() * math.sqrt(t_bin))
        t+=1
    return t * t_bin

def eq1(t,T):
    s = 2
    arr =np.sort(t, axis=None)
    
    t_bind=1
    t_next=2
    while t_next < len(arr):
        if(arr[t_next] -arr[t_bind] > T):
            s+=1; t_bind = t_next; t_next+=1
        else:
            t_next+=1
    s_arr = np.zeros((len(arr)-1,), dtype=int)
    s_arr[s-2] = 1
    return s_arr
        
def sim(N,d,v,sigma,T,tau, no_sim=100000):
    
    # simulate N particle model
    
    s_count = np.zeros(N-1)
    for i in range(1,no_sim):
        
        tauT = np.zeros((N,),dtype=float)
        tauT[0] = 0
        for i in range(1,len(tauT)):
            tauT[i] =tauT[i-1]+ np.random.exponential(tau)
            
        t = np.zeros((N,),dtype=float)
        for i in range(0,len(t)):
            t[i] =tauT[i]+ brownian(d,v,sigma)
        
        t=t-t[0] # making relatiove to 1st particle
            
        s = eq1(t,T)
        
        s_count +=s
            
    s_prob = s_count/no_sim
    res= np.zeros((5+N-1,), dtype=float)
    res[0]=d; res[1]=v; res[2]=sigma; res[3]=T; res[4]=tau;
    res[5:]=s_prob
    return(res)

def test():
    
    N=4; d=1; v=1; sigma=1; T=1; tau=1
    start = time.time()
    res = sim(N,d,v,sigma,T,tau)
    print("result : ", res, " Time: ", time.time()-start)

def main_prl(N):
    
    # run for different parameter combination
    
    d=1; v=1; sigma=1
    t_range = np.arange(0.1,4,0.1)
    tau_range = np.arange(0.5,4,0.1)
    Data = np.zeros((len(t_range)*len(tau_range),5+N-1),dtype=float)
    res = Parallel(n_jobs=16)(delayed(sim)(N,d,v,sigma,i,j) for j in tau_range for i in t_range)    
    res = np.array(res)
    col_name= ['d', 'v', 'sigma', 'T', 'tau'] + ['S'+ str(i) for i in range(2,N+1)]
    Data = pd.DataFrame(res, columns=col_name)
    Data.to_csv('dataSimulation'+str(N)+'.csv', index =False)
  
main_prl(int(sys.argv[1]))

    
    
    
