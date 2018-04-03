# -*- coding: utf-8 -*-
"""
Created on Sun Dec 03 12:35:33 2017

@author: Rocky
"""

from sklearn.mixture import GMM 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import warnings
warnings.filterwarnings("ignore")
iris = datasets.load_iris()
X_train = iris.data[0:100]
y_train = iris.target[0:100]
#X_test = iris.data[test_index]
#y_test = iris.target[test_index]


SeModel= GMM(n_components=2,covariance_type='full', init_params='wc', n_iter=100)  
SeModel.fit(X_train[:50,[0,2]])
data = X_train[60:70,[0,2]] 

inter = 1
while (inter != 0 ):
    logPro ,comPro= SeModel.score_samples(data)
    Nk =np.sum(comPro,axis=0)
    
    cov = [] 
    ''' mean '''
    mu=comPro*data
    mu_N = mu/Nk
    Com_num = len(Nk)
    sample_num , feaDim = data.shape
    '''covar'''
    for i in range(Com_num):
        cov_k = np.mat(np.zeros((int(feaDim),int(feaDim))))
        for j in range(sample_num):
            sample = data[j,:]
            sample= sample[:,np.newaxis] 
            ''' j sample , i compont'''
            cov_k+= np.matmul(sample,sample.T) * comPro[j,i]
        cov.append(cov_k/Nk[i])
    
    relevance_fac = 16    
    regular_par = Nk/(Nk+relevance_fac)     
    com_wei = SeModel.weights_
    com_mu = SeModel.means_  
    com_cov = SeModel.covars_      
    new_wei = []
    new_mu = [] 
    new_cov = []
    ''' training '''
    for i in range(Com_num):
        print(1-regular_par[i])
        new_wei.append(regular_par[i]*Nk[i]/Com_num+(1-regular_par[i]*com_wei[i]))
        new_mu.append(regular_par[i]*mu_N[i]+(1-regular_par[i])*com_mu[i])
        muTmp =  com_mu[i]
        mu_cov = np.matmul(muTmp[:,np.newaxis] , muTmp[:,np.newaxis].T)
        new_cov.append((regular_par[i]*cov[i])+ (1-regular_par[i])*(com_cov[i]+mu_cov) - mu_cov)    
    new_wei = new_wei/sum(new_wei) 

    SeModel.weights_ = np.asarray(new_wei)   
    SeModel.means_ = np.asarray(new_mu)
    SeModel.covars_ = np.asarray(new_cov,dtype=np.float64)

    inter -=1
#cov = comPro*()





