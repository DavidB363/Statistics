#!/usr/bin/env python
# coding: utf-8

# # David Brookes
# # April 2022 

# # One Factor ANOVA (Analysis of Variance)

# # Task achieved using:
# # 1. Simple python commands
# # 2. Library software

# In many situations there is a need to test the significance of differences among three or more sampling means, or equivalently to test the null hypothesis that the sample means are all equal. 
# This is where the Analysis of Variance technique can be used.
# 
# In a one-factor experiment, measurements or observations are obtained for a independent groups of 
# samples, where the number of measurements in each group is b. We speak of a treatments, each of which
# has b repetitions or replications.
# 
# Note that the independent variable is the factor - a categorical variable, whose values are the groups (e.g. A, B, C). 
# The dependent variable is the continuous variable which takes on the values of the measurements in each group.

# Example: The yield in bushels per acre of a certain variety of wheat grown in a particular 
#     type of soil treated with chemicals A, B, or C is given in the numpy array x.
#     Note that a=3 and b=4 in this case.

# (Theory and examples taken from the book Probability and Statistics - Schaum's Outline Series).

# ANOVA test assumptions:
# 
# The ANOVA test has important assumptions that must be satisfied in order for the associated p-value to be valid.
# 
# The samples are independent.
# 
# Each sample is from a normally distributed population.
# 
# The population standard deviations of the groups are all equal. This property is known as homoscedasticity.

# In[1]:


import numpy as np


# In[2]:


x = np.array([[48, 49, 50, 49], [47, 49, 48, 48], [49, 51, 50, 50]]) 
print(x)


# In[3]:


a, b = x.shape

print((a,b))


# In[4]:


# Find the treatment (row) means.

x_row_mean = np.mean(x, axis=1)
print(x_row_mean)


# In[5]:


# Find the grand mean.

x_grand_mean = np.mean(x)
print(x_grand_mean)


# In[6]:


# Find the total variation.
# (Using the efficient numpy array iterator np.nditer).
v = 0.0
for el in np.nditer(x):
    v += (el-x_grand_mean)**2
print(v)


# In[7]:


# Find the variation between treatments (rows).
# (Using the efficient numpy array iterator np.nditer).
v_b = 0.0
for el in np.nditer(x_row_mean):
    v_b += (el-x_grand_mean)**2
v_b *= b
print(v_b)


# In[8]:


# Find the variation within treatments (rows).
# (Using the efficient numpy array iterator np.nditer).
# numpy arrays x and x__row_mean can be iterated simultaneously.

v_w = 0.0
for el, m in np.nditer([x, x_row_mean.reshape(a,1)]):
    v_w += (el-m)**2

print(v_w)


# In[9]:


# A quick way to work out the variation within treatments is as follows.
# (This acts as a check for the above calculation)

v_w = v - v_b
print(v_w)


# In[10]:


# Find an unbiased estimate of the population variance using: 
# the variation between treatments under the null hypothesis of equal treatment means.

var_b = v_b/(a-1)
print(var_b)


# In[11]:


# Find an unbiased estimate of the population variance using:
# the variation within treatments.
#
# Note that this is the best estimate of the variance regardless of
# whether the null hypothesis is true or not.

var_w = v_w/(a*(b-1))
print(var_w)


# In[12]:


# Under the null hypothesis of equal tretment means the statistic
# var_b/var_w has an F distribution with (a-1), a(b-1) degrees of freedom. 
# This provides a test for the null hyothesis.

F = var_b/var_w

print(F)


# In[13]:


# To find the p value it is necessary to calculate areas
# under the F distribution curve.
# I'm going to use the SciPy library here to save time,
# rather performing numerical integration!

from scipy.stats import f
f_stat = F
dof1 = a - 1
dof2 = a*(b-1)

# p-value for 1-sided test
print('p_value = ', 1 - f.cdf(abs(f_stat), dof1, dof2))


# In[14]:


# Note the 95% point is given by:
f_95 = f.ppf(0.95, dof1, dof2)
print(f_95)

# ... and the 99% point is given by:
f_99 = f.ppf(0.99, dof1, dof2)
print(f_99)


# # Performing ANOVA One Factor Test with SciPy.stats Library

# In[15]:



from scipy import stats
results = stats.f_oneway(*x)
# Note that *x is equivalent to x[0], x[1], x[2]... i.e. the rows of x.

print("The F-statistic is %.6f and the p-value is %.6f." % results)

