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
import matplotlib.pyplot as plt


#define fibril, micelle and monomer class
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
   
class monomer:
    def __init__(self):
        self.bind = False
        self.f_alpha = True
        self.position = 0

# main simulation
class sim:
    def __init__(self,del_t=0.001, v=1, D=0.5, l=10, m=0.01):
        self.del_t=del_t
        self.v=v
        self.D=D
        self.l=l
        self.m=m
        self.sigma = np.sqrt(self.D/2)

    def doSim(self,del_G_alpha, del_G_beta):
        # parameters
        
        #Fibril and micelle copy number
        num_mol=10; num_F=10; num_M=10;
        F_binding_site=10; M_binding_site=10;
        F_list=[]; M_list=[];mol_list=[];
        for i in range(0,num_F):
            F_list.append(fibril(F_binding_site))
        for i in range(0,num_M):
            M_list.append(micelle(M_binding_site))    
        for i in range(0,num_mol):
            mol_list.append(monomer())
            
        sim=0; count_F=0; count_M=0;
        current_time=0;
        while(len(mol_list)>0):
            current_time += self.del_t
            for i in range(0, len(mol_list)):
                del_x = self.sigma*np.sqrt(self.del_t)*np.random.randn()
                kinetic_energy = 0.5*self.m*(del_x/self.del_t)**2
                if ((mol_list[i].f_alpha == True) and (kinetic_energy>del_G_beta)):
                    mol_list[i].f_alpha=not mol_list[i].f_alpha
                    del_x = (1 if del_x > 0 else -1) *self.del_t* np.sqrt((del_x/self.del_t)**2- (2*del_G_beta/self.m)) # direction??
                if((mol_list[i].f_alpha == False) and (kinetic_energy>del_G_alpha)):
                    mol_list[i].f_alpha=not mol_list[i].f_alpha
                    del_x = (1 if del_x > 0 else -1) *self.del_t* np.sqrt((del_x/self.del_t)**2- (2*del_G_alpha/self.m))
                mol_list[i].position+= self.v*self.del_t + del_x
            
            for i in range(0, len(mol_list)):
                if(mol_list[i].position > self.l): 
                    if mol_list[i].f_alpha==True:
                        if(np.random.uniform()<(num_M/(num_M + num_F))):
                            selected_M=randrange(num_M)
                            if M_list[selected_M].binding_site > 0:
                                mol_list[i].bind_M=True
                                M_list[selected_M].bind()
                                del mol_list[i]
                                count_M+=1
                            else:
                                mol_list[i].position=0
                        else:
                            mol_list[i].position=0
                
                    if mol_list[i].f_alpha==False:
                        if(np.random.uniform()<(num_F/(num_M + num_F))):
                            selected_F=randrange(num_F)
                            mol_list[i].bind_F=True
                            F_list[selected_F].bind()
                            del mol_list[i]
                            count_F+=1
                        else:
                            mol_list[i].position=0
                    else:
                        mol_list[i].position=0
                            
        print("sim number ", sim, "bind_F ", count_F, "bind_M ", count_M)
        
        #print("Fibril bind",count_F/num_mol,"Micelle bind",count_M/num_mol)
        #print(M_list[0].binding_site, F_list[0].binding_site) 
        return(count_F, count_M)

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
    

def test(i,j, num_sim=10):
    F_b=[]; F_Time =[]; F_Type=[]
    for temp in range(num_sim):
        F_b.append(sim(i,j)[0])
        F_Time.append(sim(i,j)[1])
        F_Type.append(sim(i,j)[2])    
    print(np.array(F_b).mean(), 1-np.array(F_b).mean())
    np.array(F_b).mean()
    Data_time=np.array(F_Time)
    Data_time = np.cumsum(Data_time, axis=1)
    
    Data_Type_F=np.array(F_Type)
    Data_Type_M=np.where(Data_Type_F< 0.5, 1, 0)
    Data_Type_F = np.cumsum(Data_Type_F, axis=1)
    Data_Type_M = np.cumsum(Data_Type_M, axis=1)
    
    F_bind_time=[]
    for k in range(np.amax(Data_Type_F)-2):
        F_bind_time.append(Data_time[Data_Type_F==k].mean())
        
    M_bind_time=[]
    for k in range(np.amax(Data_Type_M)-2):
        M_bind_time.append(Data_time[Data_Type_M==k].mean())

    plt.plot(F_bind_time,'g*',label='F')
    plt.plot(M_bind_time, 'ro', label='M')
    plt.xlabel('Number of molecules')
    plt.ylabel('time')
    filename= 'sample_'+str(i) + '_' + str(j)+'.pdf'
    plt.savefig(filename)
    print(Data_time.shape)
    return np.array(F_b).mean()

def main_prl():
    start=  timeit.default_timer()        
    del_G_alpha=np.arange(1.0,4.001,1)
    del_G_beta=np.arange(1.0,4.001,1)
    result= np.empty((3,len(del_G_alpha)*len(del_G_beta)))
    x = Parallel(n_jobs=8)(delayed(test)(i,j) for j in del_G_beta for i in del_G_alpha )
    #print(x)
    count=0
    for i in del_G_alpha:
            for j in del_G_beta:
                result[0,count]= i; result[1,count]=j;count+=1;
    result[2,:]=np.array(x)
    np.savetxt("result_extension_F10_M10_N200.csv",result,delimiter=",")
    print("Time: ", timeit.default_timer() - start)

a =sim()
sim.doSim(a,del_G_alpha=1, del_G_beta=1)
#main_prl()




