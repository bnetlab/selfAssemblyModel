#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 13:46:07 2019

@author: ranap
"""

import numpy as np

num_sim=10**4

#parameters
del_t=0.001; v=1; D=0.5; l=10; m=1; del_G = 1

#Fibril and micelle concentration
F=0.5; M=0.5
#loop and result variable
sim=0; count_F=0; count_M=0;

while(sim<num_sim):
    bind_M=False; bind_F=False; f_alpha=True;
    while(not(bind_F) and not(bind_M)):
        position=0
        while (position< l):
            del_x = 2*D*del_t*np.random.randn()
            v_eff = v + del_x/del_t
            kinetic_energy = 0.5*m*(del_x/del_t)**2
            if(kinetic_energy>del_G):
                f_alpha=not f_alpha
                del_x = (1 if del_x > 0 else -1) *del_t* np.sqrt((del_x/del_t)**2- (2*del_G/m)) # direction??
            position+= v*del_t + del_x
        print(f_alpha)
        
        if f_alpha==True:
            if(np.random.uniform()<(M/(M+F))):
                bind_M=True
        
        if f_alpha==False:
            if(np.random.uniform()<(F/(M+F))):
                bind_F=True
        
    count_F+=bind_F
    count_M+=bind_M
    print("sim number ", sim, "bind_F ", bind_F, "bind_M ", bind_M)
    sim+=1

print("Fibril bind",count_F/num_sim,"Micelle bind",count_M/num_sim)

        




