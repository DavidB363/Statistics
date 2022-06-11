#!/usr/bin/env python
# coding: utf-8

# # David Brookes
# # January 2022

# # The Chi Squared Statistical Test.

# # Task achieved using:
# # 1. Simple python commands
# # 2. Library software

# In[1]:


# The Chi Squared Statistical Test.
# Dependent variable is discrete. It is a set of events in the example given.
# Independent variable is discrete. Individual experiments are considered here.


# In[2]:


# Consider 5 events that can occur; A1, A2, A3, A4 and A5.
# Hypothesise that the probabilities are p1, p2, p3, p4 and p5 respectively.
# (Note that the sum of the p_i is 1.)


# In[3]:


# If an experiment is run n times and the number of observed
# occurences of the events are x1, x2, x3, x4 and x5 (sum of x_i is n), is it reasonable to assume
# that the probabilities are correct, or is there evidence that the observed frequencies are
# significantly different from those expected?


# In[4]:


import numpy as np

np.random.seed(4) # This value of the seed has been chosen in order to
                # give a statistically significant result (i.e small p value).

from numpy import random

numevents = 5 # Number of events; A1 to A5.

n =100 # The number of trials with outcome one of A1 to A5.

eventcounts = np.zeros(numevents)
probabilities = np.zeros(numevents)
expectedcounts = np.zeros(numevents)
cumulative_probs = np.zeros(numevents)
# print(eventcounts)
# print(probabilities)


# In[5]:


# Generate the probabilities p1 to p5.

probabilities = np.random.random(numevents)
# print(probabilities)

# Normalise the probabilities.
sum_probs = probabilities.sum()
# print('sum_probs', sum_probs)
probabilities = probabilities/sum_probs
sum_probs = probabilities.sum()
# print('sum_probs', sum_probs)
print('probabilities \n', probabilities)  

cumulative_probs[0] = probabilities[0]
for i in range(1,numevents):
    cumulative_probs[i] = cumulative_probs[i-1] + probabilities[i]
    
# print('cumulative_probs', cumulative_probs) 

# Calculate the expected number of counts.

expectedcounts = probabilities*n
# print('expectedcounts ', expectedcounts)
# print('eventcounts ',eventcounts)


# In[6]:


# Generate the frequencies of events; x1 to x5.

for i in range(n):
    val = np.random.random()
    for j in range(numevents):
        if val<cumulative_probs[j]:
            eventcounts[j] += 1
            #print('Event number ', j+1)
            break

print('eventcounts \n', eventcounts)
print('expectedcounts \n', expectedcounts)
#print('sum_eventcounts', eventcounts.sum())
    


# In[7]:


# Compute the chi squared statistic.

chi2 = 0.0
for i in range(numevents):
    chi2 += ((eventcounts[i]-expectedcounts[i])**2)/expectedcounts[i]
    
print('chi2 = ', chi2)


# In[8]:


# Degrees of freedom.
df = numevents-1
print('df = ', df)


# # Calculating Chi Squared using the SciPy Library

# In[9]:


# To calculate the p value, the easiest way is to use the scipy library.
# scipy.stats.chisquare calculates the chi square statistic and p value.
# Note that the number of degrees of freedom is numevents-1.

from scipy.stats import chisquare
chisq, p = chisquare(eventcounts, f_exp = expectedcounts)

print('chisq = ', chisq)
print('p = ', p)

# Note the small p value implying that the measured data
# is significantly different from that expected.


# In[ ]:




