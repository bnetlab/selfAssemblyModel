#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 13:02:14 2019

@author: ranap
4 molecule 

V = l/mu
D = 2 * sigma^2 = (2*l^2) / lambda 

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from scipy.stats import invgauss
import math
import time
import sys 

def brownian(d,v,sigma,t_bin=0.001):
    
    # simulate brownian motion
    
    pos= 0
    t= 1
    v_t = v*t_bin
    while pos < d:
        pos = pos + v_t + (sigma * np.random.randn() * math.sqrt(t_bin))
        t+=1
    return t * t_bin

def eq1s2(t2,t3,t4,T):
    arr = np.array([t2,t3,t4,0])
    arr = np.sort(arr, axis=None)
    a= 1- np.heaviside (np.absolute(arr[3] -arr[1])- T, 0)
    return a

def eq1s4(t2,t3,t4,T):
    arr = np.array([t2,t3,t4,0])
    arr = np.sort(arr, axis=None)
    a= bool(np.heaviside (np.absolute(arr[2] -arr[1])- T, 0)) and bool((np.heaviside(np.absolute(arr[3] -arr[2])- T, 0)))
    return a

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
        tauT[0] = 0;
        for i in range(1,len(tauT)):
            tauT[i] =tauT[i-1]+ np.random.exponential(1/tau);
            
        t = np.zeros((N,),dtype=float)
        for i in range(0,len(t)):
            t[i] =tauT[i]+ brownian(d,v,sigma);
        
        t=t-t[0] # making relatiove to 1st particle
            
        s = eq1(t,T)
        
        s_count +=s
            
    s_prob = s_count/no_sim
    return(s_prob)

def isIG(d,v,sigma, no_sim=100000, n_bins=20):
    
    # check the distribution is IG or not
    
    simData = np.empty([no_sim,])
    for i in range(1,no_sim):
        simData[i] = brownian(d,v,sigma)
    
    fig = plt.figure()
    
    plt.hist(simData, normed=True, bins=n_bins, range = (0,5*(d/v)))
    plt.ylabel('Prob')
    
    mu=d/v; lamda=d**2/sigma**2;
    igData =  invgauss.rvs(mu/lamda, scale=lamda, size=no_sim)

    plt.hist(igData, normed=True, histtype='step',range = (0,5*(d/v)), bins=n_bins, color='r')
    
    
    filename = 'plot_' + str(d) + '_' + str(v) +'_' + str(sigma) + '.png'
    plt.savefig(filename)
    
def main(N):
    
    # run for different parameter combination
    
    d=1; v=1; sigma=1;
    t_range = [0.5,1,1.5,2,2.5,3,3.5,4]
    tau_range = [0.5,1,1.5,2,2.5,3,3.5,4]
    count=0; Data = np.zeros((len(t_range)*len(tau_range),5+N-1),dtype=float)
    start = time.time()
    for T in t_range:
        for tau in tau_range:
            res = sim(N,d,v,sigma,T,tau)
            Data[count,0] = d; Data[count,1] = v; Data[count,2] = sigma;
            Data[count,3] = T; Data[count,4] = tau;
            for i in range(len(res)):
                Data[count,5+i] = res[i]
            count+=1;
            print("Done :", count, " Time : ",time.time() - start)
    
    pd.DataFrame(Data).to_csv('dataSimulation'+str(N)+'.csv')
    
def test():
    
    # test mode
#    for d in [1,2,4]:
#        for v in [1,2,4]:
#            for sigma in [1,0.5,0.25]:
#                isIG(d,v,sigma)
                
    N=4; d=1; v=1; sigma=1; T=1; tau=4;
    start = time.time()
    res = sim(N,d,v,sigma,T,tau)
    print("result : ", res, " Time: ", time.time()-start)
        
# main(int(sys.argv[1]))    

test()  
    
    
    
