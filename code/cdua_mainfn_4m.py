
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
    
    
    __global__ void doIndexy(float *Z, int *indexx, int *indexy, int *P1a, int *P1b, float *P4, float *P3a, float *P3b, int totalSum, int totalThread )
    { 
        
        int idx = blockIdx.x * blockDim.x + threadIdx.x; 

        if(idx >= totalThread){
            return;
        }

        float sum_a = 0;
        float sum_b = 0;
        int j = indexy[idx]; 
        for (int i = 0; i < totalSum ; i++){
           if(Z[indexx[i]+j]>0.f){
               sum_a += 0.1*0.1*0.1 * Z[indexx[i]+j] * P1a[i];
               sum_b += 0.1*0.1*0.1 * Z[indexx[i]+j] * P1b[i];
           }
           
        }; 
        //printf("%d\t%f\t%f\t%d\t%d\t%d\t%d\\n",idx,sum1,Z[0],indexx[sizeof(indexy)/sizeof(indexy[0]) - 1], indexy[0],P1[0],total);
        P3a[idx] = sum_a*P4[idx]; 
        P3b[idx] = sum_b*P4[idx];
        
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

    f2 = pd.read_csv('savedist_4d.tsv',sep=' ', squeeze=True, header=None).values

    # reading indexx and indexy
    # indexy = np.load("indexy.npy")
    # indexx = np.load("indexx.npy")
    
    
    indexy = np.zeros(Pin_point ** 3,dtype=int)
    for k in range (0,Pin_point):
        for k1 in range (0,Pin_point):
            a = [i for i in range( Z_point*Z_point*k + Z_point*k1 , Z_point*Z_point*k + Z_point*k1 + Pin_point)] 
            indexy[ k*Pin_point*Pin_point + k1*Pin_point : k*Pin_point*Pin_point + (k1+1)*Pin_point] = a

    indexy = indexy[::-1]

    indexx = np.zeros(obs_point ** 3, dtype=int)
    for k in range (0,obs_point): 
        for k1 in range (0,obs_point): 
            a = [i for i in range(Z_point*Z_point*k + k1*Z_point, Z_point*Z_point*k + k1*Z_point + obs_point )] 
            indexx[k*obs_point*obs_point + obs_point*k1 : k*obs_point*obs_point + (k1+1)*obs_point ] = a
    
    np.save('indexy', indexy)
    np.save('indexx', indexx)
    
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

    d_P4 = cuda.mem_alloc(np.float32(p4).nbytes) 
    cuda.memcpy_htod(d_P4, np.float32(p4))

    d_P3S2 = cuda.mem_alloc(np.float32(indexy).nbytes)
    cuda.memcpy_htod(d_P3S2, np.float32(np.zeros_like(indexy)))
    d_P3S4 = cuda.mem_alloc(np.float32(indexy).nbytes)
    cuda.memcpy_htod(d_P3S4, np.float32(np.zeros_like(indexy))) 

    func = mod.get_function("doIndexy")

    blocksize = 128
    gridsize = math.floor(len(indexy)/blocksize)
    func(d_Z, d_indexx, d_indexy, d_P1S2, d_P1S4,d_P4, d_P3S2, d_P3S4, np.int32(len(p1s2)),np.int32(len(p4)), block=(blocksize,1,1), grid =(gridsize,1,1))

    cuda.Context.synchronize()
    
    h_test_outs2 = np.empty_like(np.float32(p4))
    h_test_outs4 = np.empty_like(np.float32(p4))
    cuda.memcpy_dtoh(h_test_outs2, d_P3S2)
    cuda.memcpy_dtoh(h_test_outs4, d_P3S4)

    cuda.Context.synchronize()

    print("I am out of  cuda")

    p=np.empty(6)
    p[0]=mu; p[1]=lamda; p[2]=T; p[3]=tau;
    p[4] = bin_size*bin_size*bin_size*np.sum( h_test_outs2)
    p[5] = bin_size*bin_size*bin_size*np.sum( h_test_outs4)
    
    filename = "result/testfile" + str(mu) + str (lamda) +str(T) +str (tau)
    file = open(filename,"w") 
    file.write(str(p))
    file.close() 
    print("Prob: ", p, "Sec: ", time.time() - start, "inCuda ", time.time()- startC)

if __name__== "__main__":
  mainfn(float(sys.argv[1]), float(sys.argv[2]),float(sys.argv[3]),float(sys.argv[4]),float(sys.argv[5]),float(sys.argv[6]),float(sys.argv[7]),float(sys.argv[8]))

