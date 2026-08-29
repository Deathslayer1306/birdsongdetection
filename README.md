This project uses a CNN model for classification of 5 bird species, indian cuckoo, common myna, greater flamingo, asian koel and indian peafowl. 
The dataset was created using the Xeno Canto API (https://xeno-canto.org/explore/api).

The cleaned dataset was created by trimming the audio to only have the necessary audio content and to remove the long silent parts of the audio. 

The feature extraction proccess consists of extracting the MFCC (Mel Frequency Cepstral Coefficient ) of the audio samples. 
Then these features are given to a GMM (Gaussian Mixture Model) before going to the CNN layers.


