import numpy as np
import _pickle as pickle
from scipy.io.wavfile import read
from sklearn import mixture
from MFCC_extraction import Extract_Features
import warnings
warnings.filterwarnings("ignore")

train_dir   = "Audio_Dataset/"   
dest_dir = "GMM_Models/"
train_file = "TrainingDataPath.txt"        
file_paths = open(train_file,'r')

n = 1 #n: Total number of training files. At first initialize as '1',further update in for loop.
features = np.asarray(())

print("Training...\n")
for path in file_paths: 
    path=path.strip()   
    print(path)
    sr,audio=read(train_dir + path)
    vector=Extract_Features(audio,sr)
    
    if features.size==0:
        features=vector
  
    else:
        features=np.vstack((features, vector))
    
    #Dump the model to "GMM_Models/"
    if n == 23:    
        gmm = mixture.GaussianMixture(n_components = 16, max_iter = 200, covariance_type='diag',n_init = 3)
        gmm.fit(features)
        model = path.split("-")[0]+".gmm"
        pickle.dump(gmm,open(dest_dir + model,'wb'))
        print("\nGMM model dumped for model: ",model," with data points = ",features.shape,"\n") 
        features = np.asarray(())
        n=0 #Set back to zero for next person.
    n=n + 1
