# Footstep-Voice-Identification

This experiment was submitted to MiiCare (Technical test). Here, I have trained the footstep voices of PersonA and PersonB. Further, four unknown voice datas were given to test and prediction (ie, whose footstep it is)is obtained as output. 

## Requirements:
- The experiment was done on Python 3.8.10. 
- Here, I have used the dataset provided by MiiCare. If you are preparing your own dataset, save it in Audio_Dataset folder.
- Make sure to delete GMM_Models/Sample.gmm. 
- Prepare the text files ```TrainingDataPath.txt``` and ```TestingDataPath.txt```. They should have the paths to your audios. 
- Finally prepare ```MFCC_extraction.py```, ```Training.py``` and ```Testing.py```.

## Workflow:
- Collect the dataset and modfy the text file.
- Run ```python3 Training.py ``` to train the voice datas. Internally, the program that calculates MFCC delta coefficients will be called and required data will be sent back. Further, GMM models are created and dumped to GMM_Models directory.
- To train, run the file using the command: ```python3 Testing.py ```. The new voice data's coefficients will be compared with the GMM model's coefficients and the highest probable one will be the predicted output. 

## Output:
### Training:
![img](https://user-images.githubusercontent.com/51737416/138079773-f53c3a5a-ce02-4e9e-a5b7-93e86d8f5d48.png)
![img](https://user-images.githubusercontent.com/51737416/138079905-8dff3264-a156-4ff1-ac9e-016daf807f62.png)

### Testing: 
![img](https://user-images.githubusercontent.com/51737416/138080329-f2129f42-a041-47a4-a04d-8d8acec4b84f.png)
