#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


pd.__version__


# In[3]:


housing = pd.read_csv('housing.csv')


# In[4]:


housing


# In[5]:


housing.info()


# In[6]:


housing.isnull().sum()


# In[7]:


housing['ocean_proximity'].nunique()


# In[8]:


housing[housing['ocean_proximity'] == 'NEAR BAY']['median_house_value'].mean()


# In[9]:


housing['total_bedrooms'].mean()


# In[10]:


housing.fillna(537.8705525375618)


# In[11]:


housing['total_bedrooms'].mean()


# In[12]:


import numpy as np


# In[13]:


x = np.array(housing[housing['ocean_proximity'] == 'ISLAND'][['housing_median_age', 'total_rooms', 'total_bedrooms']])


# In[14]:


x


# In[15]:


xTx = x.T.dot(x)


# In[16]:


xTx


# In[17]:


xTxinv=np.linalg.inv(xTx)


# In[18]:


xTxinv


# In[19]:


y = np.array([950, 1300, 800, 1000, 1300])


# In[20]:


y


# In[21]:


w = xTxinv.dot(x.T).dot(y)


# In[22]:


w





