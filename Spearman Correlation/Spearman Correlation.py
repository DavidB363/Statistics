#!/usr/bin/env python
# coding: utf-8

# # David Brookes
# # January 2022
# # Spearman's Rank Correlation Coefficient.

# # Task achieved using:
# # 1. Simple python commands
# # 2. Library software

# In[1]:


# Determines the rank correlation between two variables.
# Instead of using precise values, or when precision is not attainable, the variables
# can be ranked using numbers 1, 2, ..., n.


# In[2]:


import numpy as np
import matplotlib.pyplot as plt


# In[3]:


# Generate x and y ranked data.

np.random.seed(0)

n = 10

x = np.random.permutation(n)+1 # Permutation of 1, 2, 3, ..n.
print(x)
y = np.random.permutation(n)+1 # Permutation of 1, 2, 3, ..n.
print(y)


# In[4]:


# Scatter plot.
plt.scatter(x, y, marker=None)


# In[5]:


# Need to calculate the summ of the squares of the differences between the ranks.

ssd =  np.dot(y - x,y - x)
# print(ssd)


# In[6]:


# Calculate Spearman's correlation coefficient.

r_rank = 1 - 6 * ssd /(n * (n*n -1))

print('Spearman\'s Rank Correlation Coefficient is ', r_rank)


# In[ ]:





# 

# # Calculating Spearman's Correlation Coefficient using the SciPy Library

# In[7]:


from scipy import stats

r_rank_scipy, p_value = stats.spearmanr(x, y)

# Note: The (two-tailed) p-value roughly indicates the probability of an uncorrelated system 
# producing datasets that have a Pearson correlation at least as extreme as the one computed 
# from these datasets.

print('Pearson\'s Correlation Coefficient is ', r_rank_scipy)
print('The two-tailed p_value is ', p_value)

