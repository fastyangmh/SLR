# -*- coding: utf-8 -*-
"""
Created on Sun Dec 03 12:35:33 2017

@author: Rocky
"""

from sklearn.mixture import GMM 
import numpy as np
from sklearn import datasets
import warnings
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
warnings.filterwarnings("ignore")
iris = datasets.load_iris()
X_train = iris.data[0:150]
y_train = iris.target[0:150]
#X_test = iris.data[test_index]
#y_test = iris.target[test_index]

SeModel= GMM(n_components=3,covariance_type='full', init_params='wc', n_iter=100)  
SeModel.fit(X_train[:50,[0,2]])
data = X_train[50:100,[0,2]] 
sc = StandardScaler()
inter = 5000
while (inter != 0 ):
    print('inter %d'%(inter))
    comPro= SeModel.predict_proba(data)
    Nk =np.sum(comPro,axis=0)
    
   
    ''' mean '''
    numData, numCompoent = comPro.shape
    '''2 is fea dim  '''
    mu_N = np.zeros((numCompoent,2))
    for i in range(int(numCompoent)):
        for idx , val in enumerate (comPro[:,i]):
            mu_N[i,:]+=val*data[idx,:]/Nk[i]
            
    cov = [] 
    Com_num = len(Nk)
    sample_num , feaDim = data.shape
    '''covar'''
    for i in range(Com_num):
        cov_k = np.mat(np.zeros((int(feaDim),int(feaDim))))
        for j in range(sample_num):
            sample = data[j,:]
            sample= sample[:,np.newaxis] 
            ''' j sample , i compont'''
            cov_k+= (np.matmul(sample,sample.T)*comPro[j,i])/Nk[i]
        cov.append(cov_k)
    
    relevance_fac = 16    
    regular_par = Nk/(Nk+relevance_fac)     
    com_wei = SeModel.weights_
    com_mu = SeModel.means_  
    com_cov = SeModel.covars_      
    new_wei = []
    new_mu = [] 
    new_cov = []
    ''' training each compont '''
    for i in range(Com_num):
        print(regular_par[i])
        new_wei.append(regular_par[i]*(Nk[i]/numData)+(1-regular_par[i]*com_wei[i]))
        _mu = (regular_par[i]*mu_N[i])+((1-regular_par[i])*com_mu[i,:])
        new_mu.append(_mu)
        muTmp =  com_mu[i]
        muTmp = muTmp[:,np.newaxis]
        _mu = _mu[:,np.newaxis]
        mu_cov = np.matmul(muTmp,muTmp.T)
        _mu_cov = np.matmul(_mu,_mu.T)
        new_cov.append((regular_par[i]*cov[i])+(1-regular_par[i])*(com_cov[i]+mu_cov)-_mu_cov)   
    ''' update '''    
    new_wei = new_wei/sum(new_wei) 
    SeModel.weights_ = np.asarray(new_wei)   
    SeModel.means_ = np.asarray(new_mu)
    SeModel.covars_ = np.asarray(new_cov,dtype=np.float64)
    inter -=1

x=SeModel.sample(100)    
plt.scatter(x[:,0],x[:,1],facecolors='none',edgecolors='r')
plt.scatter(X_train[50:100,0] ,X_train[50:100,2],facecolors='none',edgecolors='g')
plt.scatter(X_train[0:50,0] ,X_train[0:50,2],facecolors='none',edgecolors='b')
#cov = comPro*()





