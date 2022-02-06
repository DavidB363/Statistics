#!/usr/bin/env python
# coding: utf-8

# # David Brookes
# # February 2022

# # Two Sample T Test

# # Task achieved using:
# # 1. Simple python commands
# # 2. Library software

# In[1]:


# Do the mean values of two groups of continuous data differ
# signicantly from one another?
# (This assumes that both groups are taken from populations with identical values of variance).


# In[2]:


import numpy as np


# In[3]:


# Set random seed.
np.random.seed(1)


# In[4]:


# Generate two groups of random samples.

mean_sample_ideal_1 = 200
sd_sample_ideal_1 = 30
size_sample_1 = 30

mean_sample_ideal_2 = 220
sd_sample_ideal_2 = 40
size_sample_2 = 50

group_1 = np.random.normal(loc=mean_sample_ideal_1, scale=sd_sample_ideal_1, size=size_sample_1)
group_2 = np.random.normal(loc=mean_sample_ideal_2, scale=sd_sample_ideal_2, size=size_sample_2)

print('mean_sample_ideal_1 ', mean_sample_ideal_1)
print('sd_sample_ideal_1 ',sd_sample_ideal_1) # Unbiased estimate.
print('mean_sample_ideal_2 ', mean_sample_ideal_2)
print('sd_sample_ideal_2 ',sd_sample_ideal_2) # Unbiased estimate.


# In[5]:


mean_sample_1 = np.mean(group_1) # Numpy solution to calculating the mean.
sd_sample_1 = np.std(group_1, ddof=1) # Divisor n-1 i.e. n - ddof. i.e unbiased estimate of standard deviation.

mean_sample_2 = np.mean(group_2) # Numpy solution to calculating the mean.
sd_sample_2 = np.std(group_2, ddof=1) # Divisor n-1 i.e. n - ddof. i.e unbiased estimate of standard deviation.


print('mean_sample_1 ', mean_sample_1)
print('sd_sample_1 ',sd_sample_1) # Unbiased estimate.
print('mean_sample_2 ', mean_sample_2)
print('sd_sample_2 ',sd_sample_2) # Unbiased estimate.


# In[6]:


# Pool the variances.
#
# (This assumes that both groups are taken from populations with identical values of variance).

var_pooled = ((size_sample_1 - 1)*sd_sample_1**2 + (size_sample_2 - 1)*sd_sample_2**2)/(size_sample_1+size_sample_2-2)
sd_pooled = np.sqrt(var_pooled)
print('sd_pooled = ', sd_pooled)


# In[7]:


# H0, Null Hypotheis: mean_pop_1 = mean_pop_2.
# H1, Alternative Hypothesis: mean_pop_1 != mean_pop_2 (two sided).


# In[8]:


# Calculate the T statistic.
# This has a t distribution with size_sample_1 + size_sample_2 - 2 
# degrees of freedom.


T = (mean_sample_1 - mean_sample_2)/(sd_pooled * np.sqrt(1/size_sample_1 + 1/size_sample_2))

print('T = ', T)


# In[9]:


# To find the p value it is necessary to calculate areas
# under the t distribution curve.
# I'm going to use the SciPy library here to save time,
# rather performing numerical integration!

from scipy.stats import t
t_stat = T
dof = size_sample_1 + size_sample_2 - 2

# p-value for 2-sided test
print('p_value = ', 2*(1 - t.cdf(abs(t_stat), dof)))


# # Performing a Two Sample T Test with SciPy.stats Library

# In[10]:



import scipy.stats as stats

#perform two sample t-test with equal variances
two_sample = stats.ttest_ind(a=group_1, b=group_2, equal_var=True)

print("The t-statistic is %.6f and the p-value is %.6f." % two_sample)

