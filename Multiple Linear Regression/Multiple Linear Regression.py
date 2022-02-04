#!/usr/bin/env python
# coding: utf-8

# # David Brookes
# # February 2022

# # Multiple Linear Regression

# # Task achieved using:
# # 1. Simple python commands
# # 2. Library software

# In[1]:


# Determines the best linear relationship between variables y and x_1, x_2,..., x_p.
# in the least squares sense.
# Assume x_i are the independent varaiables and y is the dependent variable.


# In[2]:


import numpy as np
import matplotlib.pyplot as plt


# In[3]:


# Generate random values for x_1, x_2,...,x_p.

# Set random seed.
np.random.seed(0)

p = 3 # Number of variables.

#n = 100 # Number of points.

n = 5

x_max = 100 # x values assumed to be between 0 and  x_max.

X =np.zeros((n,p+1)) # Note the extra column for x_0 (these are to be all set to 1.0).
                    # This is known as the augmented matrix.

# print(X)

#for i in range(p+1):
#    for j in range(n):
#        print(X[i,j])

from numpy import random

for i in range(n):
    X[i,0] = 1.0

for i in range(n):
    for j in range(1,p+1):
        X[i,j] = random.randint(0, x_max+1, size=1)

print(X)


# In[4]:


# Generate some y values using the x values from X.
# Using the formula y = a_0+a_1*x_1 + a_2*x_2 +...+a_p*x_p + random.
# i.e. linear part plus random part.


# In[5]:



# Generate the a_i randomly.

low = -10
high = 10
a = random.randint(low, high+1, size=p+1) # p+1 random integers in range [low, high].

print(a)


# In[6]:


# Generate the y values.
rand_max = 10
rand_array = np.random.rand(n) # n random floating point numbers between 0 and 1.
rand_array = rand_max*(2*rand_array -1)  # n random floating point numbers between -rand_max and rand_max.
print(rand_array)

y = np.dot(X, a) + rand_array
print(y)


# In[7]:


# Calculate the least squares best estimates for the intercept and coefficients.

XtX = np.dot(np.transpose(X), X)
XtXinv = np.linalg.inv(XtX)
a_est = np.dot(np.dot(XtXinv, np.transpose(X)), y) 

print(a_est)


# # Perform Multiple Linear Regression using the Scikit-learn Library

# In[8]:


from sklearn.linear_model import LinearRegression

X = X[:,1:] # Strip the constant 1.0 values from the augmented matrix.
            # The sklearn model does not require these.

multireg = LinearRegression().fit(X, y)

coeff_estimates  = multireg.coef_

intercept_estimate = multireg.intercept_

print('coeff_estimates =', coeff_estimates)
print('intercept_estimate =', intercept_estimate)


# In[ ]:




