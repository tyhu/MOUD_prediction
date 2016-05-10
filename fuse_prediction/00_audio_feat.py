### text feature extraction
### Ting-Yao Hu,2016.04

import sys
import glob
import string
sys.path.append('/home2/tingyaoh/ML/PyAI/util')
from util_ml import *

labfn = '../MultimodalFeatures/audioFeatures/cats.txt'
featlst = []
lablst = []
labdict = {'neutral':0,'positive':1,'negative':2}
for line in file(labfn):
    fn = '../audio_prediction/opensmile_feat/'+line.split()[0]+'.feat'
    x = file(fn).readline().split(';')
    featlst.append(x)
X = np.array(featlst).astype('float')
pickle.dump(X,open('audio.pkl','wb'))

