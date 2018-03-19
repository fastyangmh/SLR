#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 11:41:33 2018

@author: rocky
"""

import numpy as np


def DTW(x,y,type=1):
    if len(x) >= len(y):
        pair = [x,y]
    else :
        pair = [y,x]
        
    if type == 1:    
        Dist = np.zeros((len(y),len(x)))
        Op_path = np.zeros((len(y),len(x)))
        for  x_idx,x_val in enumerate(x):
            for y_idx,y_val in enumerate(y):
                Dist[y_idx,x_idx] = abs(x_val-y_val)        
        
        for i in range(Dist.shape[0]):
            for j in range(Dist.shape[1]):
                if i == 0 and j == 0:
                    Op_path[i,j]= Dist[0,0]
                elif i == 0 :
                    Op_path[i,j] = Op_path[i,j-1]+Dist[i,j]
                elif j == 0 :
                    Op_path[i,j] = Op_path[i-1,j]+Dist[i,j]
                else :
                    Op_path[i,j] = min(Op_path[i,j-1]+Dist[i,j],Op_path[i,j-1]+Dist[i,j],Op_path[i-1,j-1]+Dist[i,j]        )
        Op_Dist = np.min(Op_path[:,-1])
        return Op_Dist
            
     
#x  = np.array([1,2,5,7,8,1,5,5])
#y = np.array([8,9,8,8,8,8])
if __name__ == "__main__" :
    x = np.array([1,-3,2,9,-1,4])
    y = np.array([2,4,-1,5])
    print (DTW(x,y))
    
    
    
    
    
    