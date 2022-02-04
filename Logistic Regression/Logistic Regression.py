#!/usr/bin/env python
# coding: utf-8

# # David Brookes
# # January 2022

# # Logistic Regression Statistical Test.

# # Task achieved using:
# # 1. Simple python commands
# # 2. Library software.

# In[1]:


# Used in classification problems.
#
# Dependent variable is discrete. 
# Independent variable is continuous. 


# In[2]:


# Example.
# Assume the dependent variable is binary and discrete with labels y=0 and y=1.


# Need to maximise the log likelihood function :-
# 
# $ J(w,b)=\frac{1}{N} \sum\limits_{i=1}^{i=N}[y_i \log_{e}h(w,b,x_{i})+(1-y_i) \log_{e}(1-h(w,b,x_{i})] $
# 
# where
# $ h(w,b,x_{i}) = \frac{1}{1+e^{-(w x_{i}+b)}}$ 
# 
# which means that
# 
# $ \frac{\partial J}{\partial w} = \frac{1}{N} \sum\limits_{i=1}^{i=N} x_i(y_i -\hat y_i) $
# 
# $ \frac{\partial J}{\partial b} = \frac{1}{N} \sum\limits_{i=1}^{i=N} (y_i -\hat y_i) $
# 
# where
# $ \hat y_i = h(w,b.x_{i}) $
# 
# which can be thought of as the predicted probability of data point $x_{i}$ having the value $y=1$.
# 
# $w$ and $b$ can be simply updated using the following equations
# 
# $w_{new} = w_{old} + \alpha \frac{\partial J}{\partial w}$
# 
# $b_{new} = b_{old} + \alpha \frac{\partial J}{\partial b}$
# 
# where $\alpha$ is a small positive constant (the learning parameter).
# 
# This will ensure that the new value of $J$ will increase. (gradient ascent!).
# 
# Note that more sophisticated methods exist (such as Newton's method).

# In[3]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

# Use class LogisticRegression from file Logistic_Regression.py in current directory.
from Logistic_Regression import LogisticRegression

bc = datasets.load_breast_cancer()

# X is a 2D numpy array.
# X is a 1D numpy array.
X, y = bc.data, bc.target


# In[4]:


print(X.shape)
print(y.shape)


# In[5]:


# Split into training and test sets.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1234)


# In[6]:


# Calculate the accuracy of prediction for the binary variable y.
def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred)/len(y_true)
    return accuracy


# In[7]:


regressor = LogisticRegression(lr = 0.0001, n_iters = 1000)
regressor.fit(X_train, y_train)
y_predictions = regressor.predict(X_test)

print('Accuracy = ', accuracy(y_test, y_predictions))


# # Perform Logistic Regression using the Scikit-learn Library

# In[8]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=1.0, solver='lbfgs', max_iter=10000, multi_class='ovr')

# Use the training data to create the model.
logreg.fit(X_train, y_train)

# Make predictions.
y_preds = logreg.predict(X_test)

print('Accuracy = ', accuracy(y_test, y_preds))

