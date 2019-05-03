
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import scipy as sp
import math
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import os
import sys

import time


# In[2]:


mod = SourceModule("""
    # include <stdio.h>
    
    #define N 1000
    
    
    __global__ void doIndexy(float *Z, int *indexx, int *indexy, int *P1, int *P2, float *P3, float *P4, int total )
    { 
        
        int idx = blockIdx.x * blockDim.x + threadIdx.x; 

        if(idx >= total){
            return;
        }

        float sum1 = 0;
        float sum2 = 0;
        int j = indexy[idx]; 
        for (int i = 0; i <= total ; i++){
           if(Z[indexx[i]+j-1]>0.f){
               sum1 += 0.1*0.1*0.1 * Z[indexx[i]+j-1] * P1[i];
               sum2 += 0.1*0.1*0.1 * Z[indexx[i]+j-1] * P2[i];
           }
           
        }; 
        //printf("%d\t%f\t%f\t%d\t%d\t%d\t%d\\n",idx,sum1,Z[0],indexx[sizeof(indexy)/sizeof(indexy[0]) - 1], indexy[0],P1[0],total);
        P3[idx] = sum1; 
        P4[idx] = sum2;
        
    }""")


# In[3]:



# calculate equation 1
def eq1s2(t2,t3,t4,T):
    a=np.empty(len(t4))
    for i in range(0,len(t4)):
        arr = np.array([t2[i],t3[i],t4[i],0])
        arr = np.sort(arr, axis=None)
        a[i]=1- np.heaviside (np.absolute(arr[3] -arr[1])- T, 0)
    return a

def eq1s4(t2,t3,t4,T):
    a=np.empty(len(t4))
    for i in range(0,len(t4)):
        arr = np.array([t2[i],t3[i],t4[i],0])
        arr = np.sort(arr, axis=None)
        a[i]= bool(np.heaviside (np.absolute(arr[2] -arr[1])- T, 0)) and bool((np.heaviside(np.absolute(arr[3] -arr[2])- T, 0)))
    return a

# for N=3 
def eq4(a,b,c,tau=1):
    ps = np.empty(len(a))
    for i in range(len(a)):
        if(b[i] >= a[i]) and (c[i]>=a[i]) and (c[i] >= b[i]):
            ps[i] = (1/tau**3)*(math.exp(-1*c[i]/tau))
        else: 
            ps[i] = 0
    return ps


# In[4]:


# parameter

# In[8]:


def mainfn(bin_size, Pmin, Pmax, tmin, tau, T, mu, lamda):
    start = time.time()
    tmax= Pmax - tmin;
    Pin_point= int((Pmax-Pmin)/bin_size +1);
    obs_point=int((tmax-tmin)/bin_size + 1);
    Zmax=tmax;
    Zmin=tmin-Pmax;
    Z_point=int((Zmax-Zmin)/bin_size + 1);

    t4, t3,t2 = np.meshgrid(np.arange(tmin,tmax+0.001,bin_size), np.arange(tmin,tmax+0.001,bin_size), np.arange(tmin,tmax+0.001,bin_size));
    t2 = t2.ravel()
    t3 = t3.ravel()
    t4 = t4.ravel()

    tau4,tau3,tau2 = np.meshgrid(np.arange(Pmin,Pmax+0.001,bin_size), np.arange(Pmin,Pmax+0.001,bin_size), np.arange(Pmin,Pmax+0.001,bin_size));
    tau2 = tau2.ravel()
    tau3 = tau3.ravel()
    tau4 = tau4.ravel()

    p1s2=eq1s2(t3,t4,t2,T)
    p1s4=eq1s4(t3,t4,t2,T)

    f2 = pd.read_csv('savedist_4d.tsv',sep=' ',squeeze=True,header=None).values

    # for 4
    # indexy = np.load("indexy.npy")
    # indexx = np.load("indexx.npy")
    # indexy = indexy[::-1]
    
    
    indexy = np.zeros(Pin_point*Pin_point*Pin_point,dtype=int)
    for k in range (1,Pin_point+1):
        for k1 in range (1,Pin_point+1):
            a = [i for i in range(1+(Z_point*Z_point*(k-1))+(Z_point*(k1-1)),Pin_point+1+(Z_point*Z_point*(k-1))+(Z_point*(k1-1)))] 
            indexy[(k-1)*Pin_point*Pin_point + (k1-1)*Pin_point : (k-1)*Pin_point*Pin_point + k1*Pin_point] = a
    
    print("I am here")
    indexy = indexy[::-1]


    indexx = np.zeros(obs_point * obs_point * obs_point)

    for k in range (0,obs_point): 
        for k1 in range (0,obs_point): 
            a = [i for i in range(Z_point*Z_point*(k)+k1*(Z_point),Z_point*Z_point*(k)+k1*Z_point+obs_point)] 
            indexx[k*obs_point*obs_point+obs_point*k1:k*obs_point*obs_point+(k1+1)*obs_point] = a
#     np.save('indexy', indexy)
#     np.save('indexx', indexx)
    
    print(indexx[0],indexy[0],indexx[-1],indexy[-1],f2.shape)
    
    p4 = eq4(tau3,tau4,tau2,tau)
    
    print("I am going into cuda")
    # cuda

    startC=time.time()
    d_Z = cuda.mem_alloc(np.float32(f2).nbytes)
    cuda.memcpy_htod(d_Z, np.float32(f2))

    d_indexx = cuda.mem_alloc(np.int32(indexx).nbytes)
    cuda.memcpy_htod(d_indexx, np.int32(indexx))

    d_indexy = cuda.mem_alloc(np.int32(indexy).nbytes)
    cuda.memcpy_htod(d_indexy, np.int32(indexy))

    d_P1S2 = cuda.mem_alloc(np.int32(p1s2).nbytes) 
    cuda.memcpy_htod(d_P1S2, np.int32(p1s2))

    d_P1S4 = cuda.mem_alloc(np.int32(p1s4).nbytes) 
    cuda.memcpy_htod(d_P1S4, np.int32(p1s4))

    d_P3S2 = cuda.mem_alloc(np.float32(indexy).nbytes)
    cuda.memcpy_htod(d_P3S2, np.float32(np.zeros_like(indexy)))
    d_P3S4 = cuda.mem_alloc(np.float32(indexy).nbytes)
    cuda.memcpy_htod(d_P3S4, np.float32(np.zeros_like(indexy))) 

    func = mod.get_function("doIndexy")

    blocksize = 1
    gridsize = math.floor(len(indexy)/blocksize)
    func(d_Z, d_indexx, d_indexy, d_P1S2, d_P1S4, d_P3S2, d_P3S4, np.int32(len(p1s2)),  block=(blocksize,1,1), grid =(gridsize,1,1))

    cuda.Context.synchronize()
    h_test_outs2 = np.empty_like(np.float32(p4))
    print(h_test_outs2.shape)
    h_test_outs4 = np.empty_like(np.float32(p4))
    cuda.memcpy_dtoh(h_test_outs2, d_P3S2)
    cuda.memcpy_dtoh(h_test_outs4, d_P3S4)

    cuda.Context.synchronize()

    p=np.empty(2)
    p[0] = bin_size*bin_size*bin_size*np.sum(np.multiply( p4, h_test_outs2))
    p[1] = bin_size*bin_size*bin_size*np.sum(np.multiply( p4, h_test_outs4))
    print("Prob: ", p, "Sec: ", time.time() - start, "inCuda ", time.time()- startC)


    filename = "testfile" + str(mu) + str (lamda) +str(T) +str (tau)
    file = open(filename,"w") 
    file.write(str(p))
    file.close() 


if __name__== "__main__":
  mainfn(float(sys.argv[1]), float(sys.argv[2]),float(sys.argv[3]),float(sys.argv[4]),float(sys.argv[5]),float(sys.argv[6]),float(sys.argv[7]),float(sys.argv[8]))

