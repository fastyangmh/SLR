from flask import Flask, request
import os
import scipy.io.wavfile as wavfile
import ffmpy
from API.FeaExt import MFCC
import numpy as np
#import EE 
app = Flask(__name__)
@app.route('/', methods=['POST'])
def get_name():
    mfcc_flag=0
    mfcc_arr=np.zeros(39)
    uid = request.form['uid']
    name =  request.form['name']
    path = request.form['path']
    extension = '.wav'
#   now_path=os.getcwd()	//get path
    print ('uid: %s, name: %s, path: %s' % (uid, name,path))
    videoname = os.path.basename(path)
    os.chdir(os.path.dirname(path))
    if not os.path.isfile(os.path.dirname(path)):
        ff = ffmpy.FFmpeg(
            inputs = {path: None}, 
            outputs = {videoname+extension: None}
        )
    ff.run()
#   rewrite_path = os.getcwd()   //get path
#   print 'rewrite_path: %s' % (rewrite_path)
#   print videoname
    fs,sig = wavfile.read(videoname)
    sig = mono_detection(sig)
    frameSize = 3*441
    Overlap = 0
    Hop = frameSize - Overlap
    PairPoint = EE.VAD(sig,fs,frameSize)
    for i in PairPoint:
        #for PairPoint index 0 is start point , 1 is end point
        StartTime = i[0]*441
        EndTime = i[1]*441
        Duration = (EndTime-StartTime)/fs
        sigTemp = sig[StartTime:EndTime]
        fea = MFCC(sigTemp,fs)
        if mfcc_flag == 0:
            mfcc_arr = fea
        else:
            mfcc_arr = np.vstack((mfcc_arr,fea))
    return 'OK.'
	
if __name__ == "__main__":
    app.run(debug=True)
