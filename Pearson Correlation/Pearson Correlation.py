#!/usr/bin/env python
# coding: utf-8

# # David Brookes
# # January 2022

# # Pearson's Correlation Coefficient.

# # Task achieved using:
# # 1. Simple python commands
# # 2. Library software

# In[1]:


# Determines the amount of LINEAR correlation between two continuous variables.
# Assume x is the independent varaiable and y is the dependent variable.


# In[2]:


import numpy as np
import matplotlib.pyplot as plt


# In[3]:


# Generate some x, y data randomly.
# i.e. y = ax+b + random

np.random.seed(2)

n = 30 # Number of points.
a = 1.5 # Gradient.
b = 8.6 # Intercept.
c = 25 # Amount of error.

x = np.arange(0,n,1)
print(x)
random_values = np.random.rand(n) # Between 0 and 1.
random_values = 2*random_values - 1 # Between -1 and 1.
y = a*x + b + c*random_values

print(y)


# In[4]:


# Scatter plot.
plt.scatter(x, y, marker=None)


# In[5]:


# Calculate the mean of x, and the mean of y.
x_mean = np.mean(x)
print(x_mean)

y_mean = np.mean(y)
print(y_mean)
    


# In[6]:


# Calculate the sum of the squares from the means for x and y.

s_x = np.dot(x-x_mean,x-x_mean) # This is n*variance_x
print(s_x)

s_y = np.dot(y-y_mean,y-y_mean) # This is n*variance_y
print(s_y)

s_xy = np.dot(x-x_mean,y-y_mean) # This is n*covariance_xy
print(s_xy)


# In[7]:


# Calculate Pearson's correlation coefficient.

r = s_xy/np.sqrt(s_x*s_y)
print('Pearson\'s Correlation Coefficient is ', r)


# # Calculating Pearson's Correlation Coefficient using the Numpy Library

# In[8]:


r_numpy = np.corrcoef(x, y)
print('Pearson\'s Correlation Coefficient is ', r_numpy[0,1])


# # Calculating Pearson's Correlation Coefficient using the SciPy Library

# In[9]:


from scipy import stats

r_scipy, p_value = stats.pearsonr(x, y)

# Note: The (two-tailed) p-value roughly indicates the probability of an uncorrelated system 
# producing datasets that have a Pearson correlation at least as extreme as the one computed 
# from these datasets.

print('Pearson\'s Correlation Coefficient is ', r_scipy)
print('The two-tailed p_value is ', p_value)


# In[ ]:




