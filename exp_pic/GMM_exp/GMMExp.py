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
from MAP_GMM import gmm_map_qb


iris = datasets.load_iris()
X_train = iris.data[0:100]
y_train = iris.target[0:100]
Mapdata = iris.data[100:]
#X_test = iris.data[test_index]
#y_test = iris.target[test_index]


SeModel= GMM(n_components=2,covariance_type='diag', init_params='wc', n_iter=100)  
SeModel.fit(X_train[:,[0,2]])
Mapgmm = gmm_map_qb(Mapdata[:,[0,2]],SeModel)
