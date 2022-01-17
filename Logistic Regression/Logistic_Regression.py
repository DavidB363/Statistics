#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.001, n_iters=1000):
    # lr - learning rate.
    # n_iters - number of iterations.
    
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        # Initialise parameters.
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient ascent.
        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)
            
            dw = (1/n_samples) * np.dot(X.T, y - y_predicted)
            db = (1/n_samples) * np.sum(y - y_predicted)
            
            self.weights += self.lr*dw
            self.bias += self.lr*db
    
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        
        return y_predicted_cls

    
    def _sigmoid(self, x):
        return 1/(1+np.exp(-x))
        

