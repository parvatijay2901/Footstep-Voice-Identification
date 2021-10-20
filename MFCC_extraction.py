import numpy as np
import python_speech_features as mfcc
from sklearn import preprocessing

#Function: Calculate Delta MFCCs
def Calculate(arr):
    rows,cols=arr.shape
    delta = np.zeros((rows,20))
    n=2
    for i in range(rows):
        index=[]
        j=1
        while j<=n:
            if i-j<0:
              first=0
            else:
              first=i-j
            if i+j>rows-1:
                second = rows-1
            else:
                second=i+j 
            index.append((second,first))
            j+=1
        delta[i]=(arr[index[0][0]]-arr[index[0][1]]+(2*(arr[index[1][0]]-arr[index[1][1]])))/10
    return delta

#Extract the mfcc and delta coefficients
def Extract_Features(audio,rate):
    mfcc_feature = mfcc.mfcc(audio,rate, 0.025, 0.01,20,nfft = 1280, appendEnergy = True)    
    mfcc_feature = preprocessing.scale(mfcc_feature)
    delta = Calculate(mfcc_feature)
    combined_op = np.hstack((mfcc_feature,delta)) #Append both
    return combined_op
