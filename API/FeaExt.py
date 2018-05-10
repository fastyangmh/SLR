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
        h_mfcc_39=np.reshape(mfcc_39,(mfcc_39.shape[0]*mfcc_39.shape[1]),order='F') 
        return h_mfcc_39
    elif reType == 'M':
        return mfcc_39
# Using the Test Wav excutes MFCC function.

def superFrame():
    return 0



if __name__ == '__main__':
    rate , sig = wav.read("a-0001.wav")
    MFCC_39 = MFCC(sig,rate)
    





    #x = lb.core.stft(sig,n_fft=1024,hop_length=512)
