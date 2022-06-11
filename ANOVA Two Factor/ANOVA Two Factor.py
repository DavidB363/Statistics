#!/usr/bin/env python
# coding: utf-8

# # David Brookes
# # April 2022

# # Two Factor ANOVA (Analysis of Variance)

# # Task achieved using:
# # 1. Simple python commands
# # 2. Library software

# The analysis of variance using one factor can be extended to two (or more) factors.
# (See code the 'ANOVA One Factor').    \
# E.g. Yields per acre of four different plant crops grown on lots treated with three different types of fertilizer. In this case Yield is the dependent continuous variable, and crops and fertilizer are the independent categorical variables (factors).   
# 
# Note that it is assumed that only one measurement is taken for the Yield (in contrast to the example given in the ANOVA One Factor code where there were replications - i.e. several measurements).
# 
# 
#  

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


fertilizer = ['A', 'B', 'C']
crop = ['I', 'II', 'III', 'IV']


# In[2]:


import numpy as np

x = np.array([[4.5, 6.4, 7.2, 6.7], [8.8, 7.8, 9.6, 7.0], [5.9, 6.8, 5.7, 5.2]]) 
print(x)


# In[3]:


a, b = x.shape

print((a,b))


# In[4]:


# Find the fertilizer (row) means.

x_row_mean = np.mean(x, axis=1)
print(x_row_mean)


# In[5]:


# Find the crop (column) means.

x_col_mean = np.mean(x, axis=0)
print(x_col_mean)


# In[6]:


# Find the grand mean.

x_grand_mean = np.mean(x)
print(x_grand_mean)


# In[7]:


# Find the total variation.
# (Using the efficient numpy array iterator np.nditer).
v = 0.0
for el in np.nditer(x):
    v += (el-x_grand_mean)**2
print(v)


# In[8]:


# Find the variation between fertilizers (rows).
# (Using the efficient numpy array iterator np.nditer).
v_r = 0.0
for el in np.nditer(x_row_mean):
    v_r += (el-x_grand_mean)**2
v_r *= b
print(v_r)


# In[9]:


# Find the variation between crops (columns).
# (Using the efficient numpy array iterator np.nditer).
v_c = 0.0
for el in np.nditer(x_col_mean):
    v_c += (el-x_grand_mean)**2
v_c *= a
print(v_c)


# In[10]:


# Find the variation due to error (rows).
# (Using the efficient numpy array iterator np.nditer).
# numpy arrays x and x__row_mean can be iterated simultaneously.

v_e = 0.0
for el, m1, m2 in np.nditer([x, x_row_mean.reshape(a,1), x_col_mean.reshape(1,b)]):
    v_e += (el - m1- m2 + x_grand_mean)**2

print(v_e)


# In[11]:


# A quick way to work out the variation due to error is as follows.
# (This acts as a check for the above calculation)

v_e = v - v_r - v_c
print(v_e)


# There are two hypotheses that we may want to test   \
# H0(1): The fertilizer (row) means are equal.   \
# H0(2): The crop (column) means are equal.     

# In[12]:


# Find an unbiased estimate of the population variance using: 
# the variation due to error.

# Note that this is the best estimate of the variance regardless of
# whether either of the null hypotheses is true or not.

var_e = v_e/((a-1)*(b-1))
print(var_e)


# In[13]:


# Find an unbiased estimate of the population variance using: 
# the variation between fertilizers (rows), under the null hypothesis
# that all fertilizer (row) means are equal (i.e. H0(1) is true).

var_r = v_r/(a-1)
print(var_r)


# In[14]:


# Find an unbiased estimate of the population variance using: 
# the variation between crops (columns), under the null hypothesis
# that all crop (column) means are equal (i.e. H0(2) is true).

var_c = v_c/(b-1)
print(var_c)


# In[15]:


# Find an unbiased estimate of the population variance using: 
# the total variation, under the null hypothesis
# that all fertilizer (row) means are equal, that all crop (column) means are equal (i.e. H0(1) and H0(2) are true).

var = v/(a*b-1)
print(var)


# In[16]:


# Under the null hypothesis H0(1) of equal fertilizer (row) means the statistic
# var_r/var_e has an F distribution with (a-1), (a-1)(b-1) degrees of freedom. 
# This provides a test for the null hyothesis.

F1 = var_r/var_e

print(F1)


# In[17]:


# To find the p value it is necessary to calculate areas
# under the F distribution curve.
# I'm going to use the SciPy library here to save time,
# rather performing numerical integration!

from scipy.stats import f
f_stat = F1
dof1 = a - 1
dof2 = (a-1)*(b-1)

# p-value for 1-sided test
print('p_value = ', 1 - f.cdf(abs(f_stat), dof1, dof2))


# At the 0.05 level of signicance H0(1) can be rejected (since 0.034 < 0.05).   \
# The fertilizer (row) means are not equal, and there is a difference in yield   \
# due to the fertilizers used.

# In[18]:


# Under the null hypothesis H0(2) of equal crop (column) means the statistic
# var_r/var_e has an F distribution with (b-1), (a-1)(b-1) degrees of freedom. 
# This provides a test for the null hyothesis.

F2 = var_c/var_e

print(F2)


# In[19]:


# Calculate the corresponding p value.

f_stat = F2
dof1 = b - 1
dof2 = (a-1)*(b-1)

# p-value for 1-sided test
print('p_value = ', 1 - f.cdf(abs(f_stat), dof1, dof2))


# At the 0.05 level of signicance H0(2) cannot be rejected (since 0.512 > 0.05).
# The crop (column) means are equal, and there is no difference in yield due to the crops used.

# # Performing ANOVA Two Factor Test with statsmodels.api Library

# In[20]:


# Note that the scipy library does not appear to have a Two Factor ANOVA function.


# In[21]:


import numpy as np
import pandas as pd


# In[22]:


# Create a dataframe.
df = pd.DataFrame(columns = ['Fertilizer', 'Crop', 'Yield'])

print(df)


# In[23]:


# Populate the dataframe with data from numpy array x.
for findex in range(len(fertilizer)):
    for cindex in range(len(crop)):
        #print('findex:', findex,'cindex:', cindex )
        df.loc[len(df.index)] = [fertilizer[findex], crop[cindex], x[findex, cindex]]    
print(df)       


# In[24]:


# Importing libraries.
import statsmodels.api as sm
from statsmodels.formula.api import ols
  
# Performing Two Factor ANOVA.
#model = ols('Yield ~ C(Fertilizer) + C(Crop) + C(Fertilizer):C(Crop)', data=df).fit()
model = ols('Yield ~ C(Fertilizer) + C(Crop)', data=df).fit()
result = sm.stats.anova_lm(model, typ=2)

print(result)

