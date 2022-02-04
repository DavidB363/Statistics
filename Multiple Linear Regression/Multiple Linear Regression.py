#!/usr/bin/env python
# coding: utf-8

# # David Brookes
# # February 2022

# # Multiple Linear Regression

# In[1]:


# Determines the best linear relationship between variables y and x_1, x_2,..., x_p.
# in the least squares sense.
# Assume x_i are the independent varaiables and y is the dependent variable.


# In[2]:


import numpy as np
import matplotlib.pyplot as plt


# In[3]:


# Generate random values for x_1, x_2,...,x_p.

p = 3 # Number of variables.

#n = 100 # Number of points.

n = 5

x_max = 100 # x values assumed to be between 0 and  x_max.

X =np.zeros((p+1,n)) # Note the extra column for x_0 (these are to be all set to 1.0).

# print(X)

#for i in range(p+1):
#    for j in range(n):
#        print(X[i,j])

from numpy import random

for j in range(n):
    X[0,j] = 1.0

for i in range(1,p+1):
    for j in range(n):
        X[i,j] = x_max*random.rand()

print(X)


# In[4]:


# Generate some y values using the x values from X.
# Using the formula y = a_0+a_1*x_1 + a_2*x_2 +...+a_p*x_p + random.
# i.e. linear part plus random part.


# In[6]:


y =np.zeros(n)

# Generate the a_i randomly.
#a = np.zeros(p+1)

a = random.rand(p+1) # p+1 random numbers between 0 and 1.
a = 2*a - 1 # p+1 random numbers between -1 and 1.

print(a)

#for i in range(n):
#    y_val = 0.0
#    for j in range(1,p+1):
#        y_val+= 
        


# In[ ]:




