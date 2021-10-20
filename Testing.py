import numpy as np
import os
import _pickle as pickle
from scipy.io.wavfile import read
from MFCC_extraction import Extract_Features
import warnings
warnings.filterwarnings("ignore")

test_dir   = "Audio_Dataset/"   
model_dir = "GMM_Models/"
test_file = "TestingDataPath.txt"     

gmm_files = [os.path.join(model_dir,fname) for fname in 
              os.listdir(model_dir) if fname.endswith('.gmm')]
models    = [pickle.load(open(fname,'rb')) for fname in gmm_files]
person  = [fname.split("/")[-1].split(".gmm")[0] for fname 
              in gmm_files]
   
file_paths = open(test_file,'r')

for path in file_paths:   
	path = path.strip()   
	print("\nTesting Audio: ",path)
	sr,audio=read(test_dir + path)
	vector= Extract_Features(audio,sr)
	likelihood = np.zeros(len(models)) 
	#Get the log likelihood
	for i in range(len(models)):
		gmm= models[i]
		score = np.array(gmm.score(vector))
		likelihood[i] = score.sum() 
	
	det = np.argmax(likelihood)
	print("Footstep of ", person[det]," is detected")
