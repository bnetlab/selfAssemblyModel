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
import math
import time

def brownian(d,v,sigma,t_bin=0.01):
    pos= 0
    t= 0
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
        
def sim(d,v,sigma,T,tau, no_sim=100000):
    
    s_count = np.zeros(3)
    for i in range(1,no_sim):
        
        tau1 = np.random.exponential(tau);
        tau2 = tau1 + np.random.exponential(tau);
        tau3 = tau2 + np.random.exponential(tau);
        
        t1 = tau1 + brownian(d,v,sigma)
        t2 = tau2 + brownian(d,v,sigma)
        t3 = tau3 + brownian(d,v,sigma)
        
        s4 = eq1s4(t1,t2,t3,T)
        s2 = eq1s2(t1,t2,t3,T)
        
        if s2 == True:
            s_count[0] += 1
        elif s4 == True:
            s_count[2] += 1
        else:
            s_count[1] +=1
            
    s_prob = s_count/no_sim
    return(s_prob)
    
def main():
    d=1; v=1; sigma=1;
    count=0; Data = np.zeros((64,8),dtype=float)
    start = time.time()
    for T in [0.5,1,1.5,2,2.5,3,3.5,4]:
        for tau in [0.5,1,1.5,2,2.5,3,3.5,4]:
            res = sim(d,v,sigma,T,tau)
            Data[count,0] = d; Data[count,1] = v; Data[count,2] = sigma;
            Data[count,3] = T; Data[count,4] = tau;
            Data[count,5] = res[0]; Data[count,6] = res[1]; Data[count,7] = res[2];
            count+=1;
            print("Done :", count, " Time : ",time.time() - start)
    pd.DataFrame(Data).to_csv('dataSimulation.csv')
    

main()    
    
    
    
    