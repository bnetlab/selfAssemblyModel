
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import re


# In[2]:


pd.set_option('display.float_format', lambda x: '%.3f' % x)


# In[3]:


df= pd.DataFrame(columns=['mu', 'lambda', 'T', 'tau', 'S2', 'S4'])
import glob
import errno
path = '/home/ranap/selfAssemblyModel/code/result2/*'
files = glob.glob(path)
for name in files:
    try:
        with open(name) as f:
            data=f.read()
            da = pd.DataFrame(re.sub(' +', ' ', data[2:-2]).split(' ')).T
            if da.shape[1] == 6:
                
                da.columns = ['mu', 'lambda', 'T', 'tau', 'S2', 'S4']
                df = pd.concat([df,da])
            if da.shape[1]==7:
                da = pd.DataFrame(re.sub(' +', ' ', data[1:-1]).split(' ')).T
                if da.shape[1]==6:
                    da.columns = ['mu', 'lambda', 'T', 'tau', 'S2', 'S4']
                    df = pd.concat([df,da])
                else:
                    print(da.shape)
                    da.columns = ['dro','mu', 'lambda', 'T', 'tau', 'S2', 'S4']
                    da.drop(columns= 'dro', axis=1, inplace=True)
                    df = pd.concat([df,da])
    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise


# In[4]:


df = df.sort_values(by=['mu', 'lambda','tau', 'T'])


# In[5]:


df


# In[6]:


df.shape


# In[7]:


df.to_csv ('result_bin0.05_range_10_20.csv', index=None)

