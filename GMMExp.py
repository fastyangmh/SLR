#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 13:46:46 2018

@author: rocky
"""

from sklearn.mixture import GMM
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


iris = datasets.load_iris()
X_train = iris.data[0:50]
y_train = iris.target[0:50]
#X_test = iris.data[test_index]
#y_test = iris.target[test_index]


SeModel= GMM(n_components=2,covariance_type='diag', init_params='wc', n_iter=100)  
SeModel.fit(X_train[:,[0,2]])
data = SeModel.sample(50)

plt.scatter(X_train[:,[0]],X_train[:,[2]],label='Iris1',facecolors='none',color='b')
#plt.scatter(data[:,[0]],data[:,[1]],label='GMM-Iris1' ,facecolors='none',color='r')
plt.scatter(iris.data[50:100,0],iris.data[50:100,2],label='Iris2',facecolors='none',color='g')
plt.legend()
plt.show()
