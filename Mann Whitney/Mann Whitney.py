#!/usr/bin/env python
# coding: utf-8

# # David Brookes
# 
# # March 2022

# # Mann Whitney Test (U Test)

# # Task achieved using:
# # 1. Simple python commands
# # 2. Library software

# In[1]:


# This is a non-parametric version of the two sample t test.
# Used to determine if two samples appear to come from the same population or that they are
# significantly different.
# The data is assumed to be continuous.
#
# The procedure is as follows:
# 1. Combine all samples into one array from smallest to largest (assume there are no tied results).
# 2. Assign ranks to the data i.e. 1 (smallest), 2, 3, ... n (largest).
# 3. Find the sum of the ranks for each of the samples, denoted by R1 and R2, where N1 and N2 are
# the respective sample sizes. Choose N1 such that N1<=N2.
# 4. Calculate the test statistic:


# $U = N_1 N_2 + \frac{N_1(N_1+1)}{2} - R_1$

# In[2]:


# corresponding to sample 1.
# The sampling distribution is symmetric, and it is possible to show that the
# mean and variance of the distribution of U is given by:


# $\mu_U = \frac{N_1 N_2}{2}$ and $\sigma_U^2=\frac{N_1N_2(N_1+N_2+1)}{12}$

# In[3]:


# See the following web page for the proof of this.
# https://www.real-statistics.com/non-parametric-tests/wilcoxon-rank-sum-test/wilcoxon-rank-sum-test-advanced/


# In[4]:


# It can also be shown that for N1+N2 > 20 that U is approximately normal so that:


# $Z=\frac{U-\mu_U}{\sigma_U}$

# In[5]:


# is normally distributed with mean 0 and variance 1.
# Hence the usual tests of significance can be carried out


# In[6]:


import numpy as np


# In[7]:


# Set random seed.
np.random.seed(1)


# In[8]:


# Generate two groups of random samples.
# (This is the same data as generated in the two sample t test program).

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


# In[9]:


# Sample 1 has the smaller number of samples, therefore N1=30 and N2=50.


# In[10]:


# Calculate the rank sum.
group_3 = np.concatenate([group_1, group_2])
sort_index_3 = np.argsort(group_3)

R1 = 0
R2 = 0
for i in range(len(sort_index_3)):
    if sort_index_3[i] < len(group_1):
        R1 += i+1
    else:
        R2 += i+1
        
print('R1 = ', R1)
print('R2 = ', R2)


# In[11]:


# Calculate the U statistic.

N1 = size_sample_1
N2 = size_sample_2

U = N1*N2 + 0.5*N1*(N1+1) - R1

print('U = ', U)


# In[12]:


# Calculate mean and standard deviation of U.

mu_U =  0.5*N1*N2
std_U = np.sqrt(N1*N2*(N1+N2+1)/ 12.0)

# Calculate the Z statistic.

Z = (U-mu_U)/std_U
print('Z = ', Z)


# In[17]:


# To find the p value it is necessary to calculate areas
# under the Z distribution curve.
# I'm going to use the SciPy library here to save time,
# rather performing numerical integration!

# Find p-value for a two-tailed test.

import scipy.stats

p_value = scipy.stats.norm.sf(abs(Z))*2

print('p_value = %.6f' % p_value)


# # Performing a Mann Whitney U Test with SciPy.stats Library

# In[14]:


from scipy.stats import mannwhitneyu
mann_whitney_results = mannwhitneyu(group_2, group_1, alternative='two-sided')

print("The U statistic is %.6f and the p-value is %.6f." % mann_whitney_results)


# # ROUGH WORK BELOW

# In[15]:


# Calculating the rank sum.

import numpy as np

sample_1 = np.array([1, 9, 5, 13])
sample_2 = np.array([18, 8, 2, 10])
sample_3 = np.concatenate([sample_1, sample_2]) 

print(type(sample_1))
print('sample_1:', sample_1)
print(type(sample_2))
print('sample_2:', sample_2)
print(type(sample_3))
print('sample_3:', sample_3)
sort_index_1 = np.argsort(sample_1)
print(type(sort_index_1))
print('sort_index_1:', sort_index_1)

sort_sample_3 = np.sort(sample_3)
print('sort_sample_3:', sort_sample_3)

sort_index_3 = np.argsort(sample_3)
print('sort_index_3:', sort_index_3)


# In[16]:


R1 = 0
R2 = 0
for i in range(len(sort_index_3)):
    if sort_index_3[i] < len(sample_1):
        R1 += i+1
    else:
        R2 += i+1
print('R1 = ', R1)
print('R2 = ', R2)
    

