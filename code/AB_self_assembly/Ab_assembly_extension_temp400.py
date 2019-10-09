#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 13:46:07 2019

@author: ranap
"""

#import
import numpy as np
from joblib import Parallel, delayed
from random import randrange
import timeit

#define fibril and micelle class
class fibril:
    def __init__(self,  binding_site):
        self.binding_site = binding_site    
    def bind(self):
        self.binding_site += 1

class micelle:
    def __init__(self,  binding_site):
        self.binding_site = binding_site    
    def bind(self):
        self.binding_site -= 1

# main simulation
def sim(del_G_alpha, del_G_beta):
    # parameters
    del_t=0.001; v=1; D=0.5; l=10; m=.01;
    sigma = np.sqrt(D/2)
    
    #Fibril and micelle copy number
    num_mol=400
    num_F=10; num_M=10;
    F_binding_site=10; M_binding_site=10;
    F_list=[]; M_list=[]; 
    
    # declearing number of micelle and fibril
    for i in range(0,num_F):
        F=fibril(F_binding_site)
        F_list.append(F)
    for i in range(0,num_M):
        M=micelle(M_binding_site)
        M_list.append(M)
        
    #loop and result variable
    sim=0; count_F=0; count_M=0;
    
    while(sim<num_mol):
        bind_M=False; bind_F=False; f_alpha=True;
        while(not(bind_F) and not(bind_M)):
            position=0
            while (position< l):
                #inside channel
                del_x = sigma*np.sqrt(del_t)*np.random.randn()
                # v_eff = v + del_x/del_t
                kinetic_energy = 0.5*m*(del_x/del_t)**2
                if f_alpha == True:
                    if(kinetic_energy>del_G_beta):
                        f_alpha=not f_alpha
                        del_x = (1 if del_x > 0 else -1) *del_t* np.sqrt((del_x/del_t)**2- (2*del_G_beta/m)) # direction??
                else:
                    if(kinetic_energy>del_G_alpha):
                        f_alpha=not f_alpha
                        del_x = (1 if del_x > 0 else -1) *del_t* np.sqrt((del_x/del_t)**2- (2*del_G_alpha/m))
                position+= v*del_t + del_x
    
            # print(f_alpha)
            # end of the channel
            if f_alpha==True:
                if(np.random.uniform()<(num_M/(num_M + num_F))):
                    selected_M=randrange(num_M)
                    if M_list[selected_M].binding_site > 0:
                        bind_M=True
                        M_list[selected_M].bind()
            
            if f_alpha==False:
                if(np.random.uniform()<(num_F/(num_M + num_F))):
                    selected_F=randrange(num_F)
                    bind_F=True
                    F_list[selected_F].bind()
            
        count_F+=bind_F
        count_M+=bind_M
        #print("sim number ", sim, "bind_F ", bind_F, "bind_M ", bind_M)
        sim+=1
    
    #print("Fibril bind",count_F/num_mol,"Micelle bind",count_M/num_mol)
    #print(M_list[0].binding_site, F_list[0].binding_site) 
    return(count_F/num_mol)

def main():
    del_G_alpha=np.arange(0.5,4.001,0.5)
    del_G_beta=np.arange(0.5,4.001,0.5)
    result= np.empty((4,len(del_G_alpha)*len(del_G_beta)))
    count=0;
    for i in del_G_alpha:
        for j in del_G_beta:
            F_b , M_b = sim(i,j)
            result[0,count]= i; result[1,count]=j; 
            result[2,count]=F_b ; result[3,count]= M_b;
            count+=1
            print(count)
    np.savetxt("result.csv",result,delimiter=",")

def test(i,j, num_sim=10**3):
    F_b=[]
    for temp in range(num_sim):
        F_b.append(sim(i,j))
    print(np.array(F_b).mean(), 1-np.array(F_b).mean())
    return np.array(F_b).mean()

def main_prl():
    start=  timeit.default_timer()        
    del_G_alpha=np.arange(0.5,4.001,0.5)
    del_G_beta=np.arange(0.5,4.001,0.5)
    result= np.empty((3,len(del_G_alpha)*len(del_G_beta)))
    x = Parallel(n_jobs=16)(delayed(test)(i,j) for j in del_G_beta for i in del_G_alpha )
    #print(x)
    count=0
    for i in del_G_alpha:
            for j in del_G_beta:
                result[0,count]= i; result[1,count]=j;count+=1;
    result[2,:]=np.array(x)
    np.savetxt("result_extension_F10_M10_N400.csv",result,delimiter=",")
    print("Time: ", timeit.default_timer() - start)

#test(1.0,1.0)
main_prl()




