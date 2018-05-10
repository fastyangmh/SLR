#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 14:06:46 2018

@author: rocky
"""
import scipy.io.wavfile as wav
import librosa  as lb
import numpy as np 
# MFCC frame size should be decided by audio segmentions size
# but now MFCC Size is using fixed size 
def MFCC(sig,rate,n_mfcc=13,hop_length=256,n_fft=1024,reType='M'):
    mfcc=lb.feature.mfcc(sig,rate,n_mfcc=n_mfcc,hop_length=hop_length,n_fft=n_fft)
    mfcc_delta = lb.feature.delta(mfcc)
    mfcc_delta_delta = lb.feature.delta(mfcc_delta)
    mfcc_39=np.vstack([mfcc, mfcc_delta,mfcc_delta_delta])
    if reType == 'S':
        '''
        retuen 
        MFCC one line by time sequence 
        '''
        h_mfcc_39=np.reshape(mfcc_39,(mfcc_39.shape[0]*mfcc_39.shape[1]),order='F') 
        return h_mfcc_39
    
    elif reType == 'M':
        '''
        retuen 
        MFCC Matrix 
        (NumOfFrame,feature)
        
        '''
        return mfcc_39.T

# Using the Test Wav excutes MFCC function.

def dataFrameToSuperFrame(dataFrame,stackSize=25):
    '''
    X: 
    given key:Value  and stackSize 
    return stack feature Matrix 
    '''
    return 0 

def MatrixToSuperFrame(X,stackSize=25):
    '''
    given feature Matrix X and stackSize 
    X row : feature  , X col : sample/frame  
    
    X Matrix : ( n_sample,feature )   <---   future whould be fixed
    return stack feature Matrix 
    stack feature Matrix : (n_samples, n_features)                 
    '''
    #can add numpy condiction 
    row,col = X.shape 
    if row >= stackSize:
        inter = row/stackSize
        X = X.flatten('C')
        split = np.asarray(np.split(X,inter))
    return split

def frame_padding(mfccMatrix,frameStack=25):
    ''' 
    given mfcc matrix and based on frameStack to pending or dropout
    MFCC Matrix (feature , n_sample ) <---   future whould be fixed

    return Matrix 
    '''
    row,col = mfccMatrix.shape
    numofpadd = row%frameStack
    if numofpadd != 0 :
        if numofpadd < abs(frameStack/2):
            return mfccMatrix[:row-numofpadd,:]
        else:                
            zerosMat = np.zeros((frameStack-numofpadd,col))
            return np.append(mfccMatrix,zerosMat,axis=0)
        
    return  mfccMatrix   

if __name__ == '__main__':
    rate , sig = wav.read("a-0001.wav")
    sig = sig[:4414] 
    MFCC_39 = MFCC(sig,rate)
    MFCC_39_padding = frame_padding(MFCC_39 ,frameStack=25)
    superFrame = MatrixToSuperFrame(MFCC_39_padding,stackSize=25)
 #   e_p = MatrixToSuperFrame(e,3)


