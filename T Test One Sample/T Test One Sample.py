#!/usr/bin/env python
# coding: utf-8

# # David Brookes
# # February 2022

# # One Sample T Test

# # Task achieved using:
# # 1. Simple python commands
# # 2. Library software

# In[1]:


# A population has mean mean_pop for a particular continuos variable.
# The standard deviation of the population is unknown.
# Given a sample of continuous data, can one determine whether it is likely
# to have come from the population?


# In[2]:


import numpy as np


# In[3]:


# Set random seed.
np.random.seed(1)


# In[4]:


# Assign population mean.
mean_pop = 100


# In[5]:


# Generate random sample.

mean_sample_ideal = 110
sd_sample_ideal = 20
size_sample = 30

x = np.random.normal(loc=mean_sample_ideal, scale=sd_sample_ideal, size=size_sample)


# In[6]:


def sum(a):
    s = 0
    for i in range(len(a)):
        s += a[i]
    return(s)
    
def sample_mean(a):
    return (sum(a)/len(a))

def sample_var(a):
    n = len(a)
    ss = 0
    for i in range(n):
        ss += a[i]**2
    s_var = (ss - n*sample_mean(a)**2)/(n-1)
    return (s_var)

def sample_std(a):
    return(np.sqrt(sample_var(a)))
    
mean_sample = sample_mean(x)
sd_sample = sample_std(x)
print('mean_sample = ', mean_sample)
print('sd_sample = ', sd_sample)


# In[7]:


# Note that these values are readily available using Numpy.

mean_sample = np.mean(x) # Numpy solution to calculating the mean.
sd_sample_1 = np.std(x) # Divisor n.
sd_sample = np.std(x, ddof=1) # Divisor n-1 i.e. n - ddof. i.e unbiased estimate of standard deviation.

print('mean_sample ', mean_sample)
print('sd_sample_1 ',sd_sample_1) # Biased estimate.
print('sd_sample ',sd_sample) # Unbiased estimate.


# In[8]:


# H0, Null Hypotheis: mean_sample = mean_pop.
# H1, Alternative Hypothesis: mean_sample != mean_pop (two sided).


# In[9]:


# Calculate the T statistic. 
# This has a t distribution with size_sample-1 degrees of freedom.

T = (mean_sample - mean_pop)/(sd_sample/np.sqrt(size_sample))

print('T = ', T)


# In[10]:


# To find the p value it is necessary to calculate areas
# under the t distribution curve.
# I'm going to use the SciPy library here to save time,
# rather performing numerical integration!

from scipy.stats import t
t_stat = T
dof = size_sample - 1

# p-value for 2-sided test
print('p_value = ', 2*(1 - t.cdf(abs(t_stat), dof)))


# # Performing One Sample T Test with SciPy.stats Library

# In[11]:


# 1-sample t-test.
from scipy import stats

one_sample = stats.ttest_1samp(x, mean_pop)

print("The t-statistic is %.6f and the p-value is %.6f." % one_sample)

