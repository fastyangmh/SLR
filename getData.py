#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 15:01:22 2018

@author: rocky
"""
from scipy.io import wavfile   
import API.FeaExt as FeaPro
import numpy as np
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.layers import Dense , Dropout
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from keras.utils import np_utils
dbaddr = 'db.txt'
data_addr = open(dbaddr,'r') 
dataFrame = {}
frameStack = 25
InputNode = frameStack * 39
while(True):
    line = data_addr.readline()
    line = line.replace('\n','')
    if not line:
        break 
    Labeldata = open(line+'/epd.txt')
    while(True):
        data = Labeldata.readline()
        if not data:
            break 
        ''' data[0] = addr , data[1]=label '''
        data = data.replace('\n','').split(' ')
        #print(line+'/'+data[0])
        if data[1] != '-1':
            fs,sig = wavfile.read(line+'/'+data[0])
            MFCC=FeaPro.MFCC(sig,fs,hop_length=768,reType='M')
            '''pandding zeros  or dropout frames '''
            row,col = MFCC.shape
            numofpadd = col%frameStack
            if numofpadd != 0 :
                if numofpadd < abs(frameStack/2):
                    MFCC = MFCC[:,:col-numofpadd]
                else:                
                    zerosMat = np.zeros((row,frameStack-numofpadd))
                    MFCC = np.append(MFCC,zerosMat,axis=1)
                    
            dataFrame[data[0]] = { "MFCC":MFCC, "label":data[1]}
                
keyList = list(dataFrame)
X=[]
Y=[]
cout = 0
for addr in keyList :
    print (addr)
    fea=dataFrame[addr]['MFCC']
    label = dataFrame[addr]['label']
    row,col = fea.shape
    if label == '0':
        cout+=1
        
    if col >=frameStack:
        inter = col/frameStack
        fea = fea.flatten('F')
        split = np.split(fea,inter)
        for val in split:
            Y.append(int(label))
            X.append(val)
    else :
        print('this wav less than frameStack')
            
X = np.asarray(X)
Y = np.asarray(Y).T
sc = StandardScaler()
sc.fit(X)
X_std =sc.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_std ,Y,test_size=0.3,random_state=42)
y_train_one = np_utils.to_categorical(y_train)
y_test_one = np_utils.to_categorical(y_test)
model = Sequential()
model.add(Dense(units=1024,input_dim=975,kernel_initializer='uniform',activation='sigmoid'))
model.add(Dense(units=1024,kernel_initializer='uniform',activation='sigmoid'))

model.add(Dense(units=2,kernel_initializer='uniform',activation='sigmoid'))
model.compile(loss='mse',optimizer='SGD', metrics=['accuracy'])

model.fit(x=X_train,y=y_train_one,epochs=100,validation_split=0.1)
all_pro = model.predict(X_test)
#all_pro = np.where(all_pro>0.5,1,0)
acc = accuracy_score(all_pro,y_test_one)

#df=pd.DataFrame=(dataFrame)
