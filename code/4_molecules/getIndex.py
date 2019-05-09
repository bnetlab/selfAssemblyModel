import numpy as np
import pandas as pd
import scipy as sp
import math

import os
import sys
import time

def mainfn(bin_size, Pmin, Pmax, tmin):
    start = time.time()
    tmax= Pmax - tmin;
    Pin_point= int((Pmax-Pmin)/bin_size +1);
    obs_point=int((tmax-tmin)/bin_size + 1);
    Zmax=tmax;
    Zmin=tmin-Pmax;
    Z_point=int((Zmax-Zmin)/bin_size + 1);

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

if __name__== "__main__":
  mainfn(float(sys.argv[1]), float(sys.argv[2]),float(sys.argv[3]),float(sys.argv[4]))