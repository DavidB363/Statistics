#!/usr/bin/env python
# coding: utf-8

# # David Brookes
# # January 2022

# # The Fisher Exact Statistical Test.

# # Task achieved using:
# # 1. Simple python commands
# # 2. Library software.

# In[1]:


# Example based on Fisher's original experiment to determine the probability
# of being able to determine whether tea was added before milk or vice versa
# in a series of trials.


# In[2]:


import numpy as np

import math as m


# In[3]:


# Contingency data table (2x2).

a = 1
b = 8
c = 13
d = 4

n = a+b+c+d

tea_data_original = np.array([[a,b],[c,d]])
tea_data = tea_data_original.copy() # Noe: deep copy of array.

print(tea_data)


# In[4]:


# Note that the expected values given random guessing are given below.
a_exp = (a+b)*(a+c)/n
b_exp = (a+b)*(b+d)/n
c_exp = (a+c)*(c+d)/n
d_exp = (b+d)*(c+d)/n

tea_data_expected = np.array([[a_exp,b_exp],[c_exp,d_exp]])

print(tea_data_expected)


# In[5]:


def comb(n, r): # N choose r
    return (int(m.factorial(n)/(m.factorial(n-r)*m.factorial(r))))

def hypergeometric(n, a, b, c, d):
    return (comb(a+b, a)*comb(c+d, c)/comb(n, a+c))


# In[6]:


# Calculate the probability of the getting the tea_data by random.

p_cutoff = hypergeometric(n, a, b, c, d)
print('p_cutoff = ', p_cutoff)


# In[7]:



# Calculate the minimum of the elements and margin sums.

min_value = np.min(tea_data)
print('min_value = ', min_value)
argmin_value = np.argmin(tea_data)
print('argmin_value = ', argmin_value)


margin = [a+b, c+d, a+c, b+d]
min_margin = min(margin)
print('min_margin ', min_margin)


# In[8]:


# Calculate the probabilities of all possible configurations.
# Those probabilities less than or equal to p_cutoff are then
# summed to give the p value.
#
# Need to set the element in the tea_data 2D array that has the min value.
# Values will range from 0 to min_margin for this element.
# The other 3 elements can be determined from this.
#
# First set the minimum element to zero, and adjust other elements accordingly.

index = argmin_value
#print('index = ', index)
row_min = index//2 # Row number of the minimum element.
col_min = index%2  # Column number of the minimum element.

def change_table(delta, index):

    multiplier = 1
    zero_or_one_old = None
    for i in range(0,4):
        #index = (index+1)%4
        #print('index = ', index)
        row = index//2
        col = index%2
        #print('row', row)
        #print('col', col)
        zero_or_one = (row+col)%2
        #print('zero_or_one', zero_or_one)
        if zero_or_one_old != None and zero_or_one != zero_or_one_old:
            multiplier *= -1
        #print('multiplier = ', multiplier)
        tea_data[row, col] += multiplier*delta
        zero_or_one_old = zero_or_one
        index = (index+1)%4
        #print()
        
change_table(-min_value, argmin_value)
    
print(tea_data)


# In[9]:


# Calculate the probabilities of all possible configurations.
# Those probabilities less than or equal to p_cutoff are then
# summed to give the p value.
#

p_value = 0.0
for el in range(min_margin+1):
    print(tea_data) 
    a = tea_data[0,0]
    b = tea_data[0,1]
    c = tea_data[1,0]
    d = tea_data[1,1]
    p = hypergeometric(n, a, b, c, d)
    print('p = ', p)
    if (p<= p_cutoff):
        p_value += p
    change_table(1,  argmin_value)
    print()
    
print('p_value is ', p_value)
    


# # Perform Fisher's Exact Test using the SciPy Library

# In[10]:


import scipy.stats as stats

odds_ratio, p_value2 = stats.fisher_exact(tea_data_original, alternative='two-sided')

print('p_value  is ', p_value2) 

