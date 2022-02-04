#!/usr/bin/env python
# coding: utf-8

# # David Brookes
# # February 2022

# # Simple Linear Regression

# # Task achieved using:
# # 1. Simple python commands
# # 2. Library software

# In[1]:


# Determines the best linear relationship between two variables
# in the least squares sense.
# Assume x is the independent varaiable and y is the dependent variable.


# In[2]:


# A lot of the code from Pearson Correlation code has been reused here.


# In[3]:


import numpy as np
import matplotlib.pyplot as plt


# In[4]:


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


# Calculate the best estimate of the coefficient b.

a_est = s_xy/s_x
print(a_est)


# In[8]:


# Calculate the best estimate of the coefficient a.

b_est = y_mean - a_est * x_mean
print(b_est)


# In[9]:


# Create scatter plot.
# Plot least squares line.

plt.plot(x, y, 'o', color = 'green')
plt.plot(x, a_est*x+b_est, color = 'red')


# In[10]:


print('a_est =', a_est)
print('b_est =', b_est)


# # Perform Simple Linear Regression using the Scikit-learn Library

# In[11]:


from sklearn.linear_model import LinearRegression

X = x.reshape(-1, 1) # Convert 1D array to 2D array

linreg = LinearRegression().fit(X, y)

a_estimate  = linreg.coef_

b_estimate = linreg.intercept_

print('a_estimate =', a_estimate[0])
print('b_estimate =', b_estimate)


# In[ ]:




