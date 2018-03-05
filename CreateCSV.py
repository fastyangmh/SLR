import pandas  as pd 
import scipy.io.wavfile as wav
import librosa  as lb
import numpy as np
from sklearn.decomposition import PCA 
np.set_printoptions(threshold=np.nan)
def getlabel():
    label= []
    counter =0
    for i in range(1000):
        label.append(counter)
        if i%100 ==0 and i!=0:
            counter+=1
    return label

adjust_range = 1
classData_Num = 1000
#using zfill to do zero pading 
Num_class_str = [ str(i).zfill(4) for i in range(1,classData_Num+adjust_range)]
#Name_class = ['a','b','c','d','e','f','g','h','i','j']
Name_class = ['a']
DB_dir = '/home/rocky/workspace/python/DeepLearing/speech_number_reg/DB/speechdata/my_training/'
label_index=getlabel()
#create dataframe to store training data 
train_set = pd.DataFrame()
#MFCC 39 dim extraion and apped 
print('Extration MFCC')

for name in Name_class:
    for idx, nb in enumerate (Num_class_str):
        rate,sig=wav.read(DB_dir+name+'-'+nb+'.wav')
        mfcc=lb.feature.mfcc(sig,rate,n_mfcc=13,hop_length=512,n_fft=1024)
        mfcc_delta = lb.feature.delta(mfcc)
        mfcc_delta_delta = lb.feature.delta(mfcc_delta)
        mfcc_39=np.vstack([mfcc, mfcc_delta,mfcc_delta_delta])
        h_mfcc_39=np.reshape(mfcc_39,(mfcc_39.shape[0]*mfcc_39.shape[1]),order='F') 
        train_set=train_set.append([[name+'-'+nb+'.wav',list(h_mfcc_39),label_index[idx]]])

train_set.columns=['Wav_Name','MFCC_39','Label']
train_set.to_csv('training_set_1000.csv',index=False)
print('done')








