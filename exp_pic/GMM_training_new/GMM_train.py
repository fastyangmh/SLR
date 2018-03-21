#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 13:47:28 2018

@author: rocky
"""

from sklearn import mixture

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


iris = datasets.load_iris()
X_train = iris.data[0:50]
y_train = iris.target[0:50]
#X_test = iris.data[test_index]
#y_test = iris.target[test_index]

iterArr = [100,200,400,600,800,1000]
m1=[]
m2=[]
co1=[]
co2=[]
for val in iterArr:
    SeModel=  mixture.GaussianMixture(n_components=2,covariance_type='diag',max_iter=val)  
    SeModel.fit(X_train[:,[0,2]])
    data,label = SeModel.sample(50)
    m1.append(SeModel.means_)
    co1.append (SeModel.covariances_)
    NewModel =  mixture.GaussianMixture(n_components=2,covariance_type='diag',max_iter=val)  
    NewModel.fit(data)
    m2.append (NewModel.means_)
    co2.append(NewModel.covariances_)
    






