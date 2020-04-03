#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 13:46:07 2019

@author: ranap
"""

#import
import numpy as np
from scipy.optimize import curve_fit
from joblib import Parallel, delayed
from random import randrange
import timeit
import matplotlib.pyplot as plt
import sys
import os


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
    """ Main simulation class
    
    """
    
    def __init__(self,del_t=0.01, v=1, D=0.5, l=10, m=0.01,  num_mol=100, num_F=10,num_M=10,F_binding_site=10,M_binding_site=10, add_Monomer = False):
        self.del_t=del_t
        self.v=v
        self.D=D
        self.l=l
        self.m=m
        self.sigma = np.sqrt(self.D/2)
        self.num_mol = num_mol
        self.num_F = num_F
        self.num_M = num_M
        self.F_binding_site= F_binding_site
        self.M_binding_site = M_binding_site
        self.add_Monomer = add_Monomer

    def doSim(self,del_G_alpha, del_G_beta):
        
        """This method perform the simulation
        
        Args: 
            del_G_alpha(float) : binding energy to alpha
            del_G_beta(float): binding energy to beta
        
        Returns:
            probabity of bind_F and bind_M
        
        """
    # initializing monomer, fibril and micelle
        F_list=[]; M_list=[];mol_list=[];
        for i in range(0,self.num_F):
            F_list.append(fibril(self.F_binding_site))
        for i in range(0,self.num_M):
            M_list.append(micelle(self.M_binding_site))    
        for i in range(0,self.num_mol):
            mol_list.append(monomer())
            
        count_F=0; count_M=0;
        current_time=0; time_step=0;
        bind_time =[]; bind_type=[];
        while(len(mol_list)>0 or time_step<20000):
            time_step+=1
            current_time += self.del_t
            if self.add_Monomer:
                if (time_step==20000):
                    print(self.num_mol, 'number of monomer added')
                    for i in range(0,self.num_mol):
                        mol_list.append(monomer())
            # calculation in every time step
            for i in range(0, len(mol_list)):
                del_x = self.sigma *np.sqrt(self.del_t) *np.random.randn()
                kinetic_energy = 0.5* self.m * (del_x/self.del_t)**2
                if (mol_list[i].f_alpha == True): 
                    if(kinetic_energy>del_G_beta):
                        mol_list[i].f_alpha=not mol_list[i].f_alpha
                        del_x_new = (1 if del_x > 0 else -1) * self.del_t * np.sqrt((del_x/self.del_t)**2 - (2*del_G_beta/self.m) + sys.float_info.epsilon) # direction??
                        del_x =del_x_new
                else:
                    if(kinetic_energy>del_G_alpha):
                        mol_list[i].f_alpha=not mol_list[i].f_alpha
                        del_x_new = (1 if del_x > 0 else -1) * self.del_t* np.sqrt((del_x/self.del_t)**2 - (2*del_G_alpha/self.m) + sys.float_info.epsilon)
                        del_x=del_x_new
                mol_list[i].position+= self.v*self.del_t + del_x
                
            i=0
            while(i<len(mol_list)):
                # calculation after end of the channel
                if(mol_list[i].position > self.l): 
                    if mol_list[i].f_alpha==True:
                        if(np.random.uniform()<(self.num_M/(self.num_M + self.num_F))):
                            selected_M=randrange(self.num_M)
                            if M_list[selected_M].binding_site > 0:
                                mol_list[i].bind_M=True
                                M_list[selected_M].bind()
                                del mol_list[i]
                                count_M+=1
                                bind_time.append(current_time)
                                bind_type.append(0)
                                #print('Time: ',current_time, 'mol_in_system :', len(mol_list),'bind_M_Total ',count_M )
                            else:
                                mol_list[i].position=0
                                i+=1
                        else:
                            mol_list[i].position=0
                            i+=1
                
                    elif mol_list[i].f_alpha==False:
                        if(np.random.uniform()<(self.num_F/(self.num_M + self.num_F))):
                            selected_F=randrange(self.num_F)
                            mol_list[i].bind_F=True
                            F_list[selected_F].bind()
                            del mol_list[i]
                            count_F+=1
                            bind_time.append(current_time)
                            bind_type.append(1)
                            #print('Time: ',current_time, 'mol_in_system :', len(mol_list),'bind_F_Total ',count_F)
                        else:
                            mol_list[i].position=0
                            i+=1
                    else:
                        mol_list[i].position=0
                        i+=1
                else:
                    i+=1
                            
        return count_F/(2*self.num_mol),count_M/(2*self.num_mol), bind_time, bind_type
        


def GetSimulationResult():
    """
    calculate F_binding and M_binding for different G_alpha and G_beta combination
    """
    del_G_alpha=np.arange(0.5,4.001,0.5)
    del_G_beta=np.arange(0.5,4.001,0.5)
    result= np.empty((4,len(del_G_alpha)*len(del_G_beta)))
    count=0;
    for i in del_G_alpha:
        for j in del_G_beta:
            F_b , M_b = doSimAll(i,j)
            result[0,count]= i; result[1,count]=j; 
            result[2,count]=F_b ; result[3,count]= M_b;
            count+=1
            print(count)
    np.savetxt("result.csv",result,delimiter=",")
    

def GetSimulationResult_loop():
    """
    parallel version of GetSimulationResult
    """
    start=  timeit.default_timer()        
    del_G_alpha=np.arange(1.0,4.001,1)
    del_G_beta=np.arange(1.0,4.001,1)
    result= np.empty((3,len(del_G_alpha)*len(del_G_beta)))
    x = Parallel(n_jobs=8)(delayed(doSimAll)(i,j) for j in del_G_beta for i in del_G_alpha )
    #print(x)
    count=0
    for i in del_G_alpha:
            for j in del_G_beta:
                result[0,count]= i; result[1,count]=j;count+=1;
    result[2,:]=np.array(x)
    np.savetxt("result_extension_F10_M10_N100.csv",result,delimiter=",")
    print("Time: ", timeit.default_timer() - start)


#main_prl()

def doSimAll(i, j, num_sim=100):
    F_M=[]; F_b=[]; F_Time=[]; F_Type=[];
    for temp in range(num_sim):
        a=sim()
        res= a.doSim(i,j)
        F_M.append(res[0])
        F_b.append(res[1])
        F_Time.append(res[2])
        F_Type.append(res[3])    
    print(np.array(F_M).mean(), np.array(F_b).mean())
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
    plt.savefig(os.path.join('plot',filename))
    return  np.array(F_b).mean()


def getDataAtTime(x, a, b, mapping_constant):
    
    """
    F_binding number at a given time
    
    Args:
        x time
        a G_alpha
        b G_beta
        mapping_constant
    Return:
        res F_binding number
    """
    print(x, a, b, mapping_constant)
    num_sim=10
    F_M=[]; F_b=[]; F_Time=[]; F_Type=[];
    for temp in range(num_sim):
        simulation= sim()
        res= simulation.doSim(a,b)
        F_M.append(res[0])
        F_b.append(res[1])
        F_Time.append(res[2])
        F_Type.append(res[3])    
    print(np.array(F_M).mean(), np.array(F_b).mean())
    Data_time=np.array(F_Time)
    Data_time = np.cumsum(Data_time, axis=1)

    Data_Type_F=np.array(F_Type)
    Data_Type_M=np.where(Data_Type_F< 0.5, 1, 0)
    Data_Type_F = np.cumsum(Data_Type_F, axis=1)
    Data_Type_M = np.cumsum(Data_Type_M, axis=1)
    print(Data_Type_F)
    print(Data_Type_M)

    F_bind_time=[0]
    for k in range(np.amax(Data_Type_F)-2):
        F_bind_time.append(Data_time[Data_Type_F==k].mean())

    M_bind_time=[0]
    for k in range(np.amax(Data_Type_M)-2):
        M_bind_time.append(Data_time[Data_Type_M==k].mean())
    #print(k* sum(np.array(F_bind_time)<x))
    res= []
    for i in x:
        res.append(mapping_constant * sum(np.array(F_bind_time)<i))
    print(F_bind_time)
    print(res)
    return res

def fitting():
    """
    Fit the model with expremental data
    
    Returns:
        estimated G_alpha, G_beta, mapping constant
    """
    # dummay data
    x = np.random.uniform(0., 10000., 100)
    y = 3. * x + 2. + np.random.normal(0., 10., 100)
    # fitting using Levenberg-Marquardt algorithm
    popt, pcov = curve_fit(getDataAtTime, x, y, bounds=(0, [2., 2., 100.]))
    print(popt)

fitting()
#getDataAtTime([50],2.,2.,100)