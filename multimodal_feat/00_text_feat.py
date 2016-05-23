### text feature extraction
### Ting-Yao Hu,2016.04

import sys
import glob
import string
from myconfig import *
sys.path.append(util_dir)
from util_ml import *
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import cPickle as pickle

text_dict = {}
duration_dict = {}
text_dir = '../VideoReviews/transcriptions/'
textfnlst = glob.glob(text_dir+'*.csv')
for fn in textfnlst:
    infile = file(fn)
    infile.readline()
    count=0
    key = fn.split('/')[-1].split('.')[0]
    for line in infile:
        count+=1
        key_utt = key+'_'+str(count)
        dur = float(line.split(';')[1])-float(line.split(';')[0])
        duration_dict[key_utt] = dur
        raw_text = line.split(';')[2][1:-1]
        raw_text = "".join([ch for ch in raw_text if ch not in string.punctuation])
        raw_text = raw_text.lower()
        text_dict[key_utt] = raw_text
pickle.dump(duration_dict,open('duration.pkl','wb'))
pickle.dump(text_dict,open('text_dict.pkl','wb'))

### unigram dictionary
unigramlst = []
for key, item in text_dict.iteritems():
    lst = item.split()
    unigramlst+=lst
unigram_set = list(set(unigramlst))
unigram_idx = {}
for idx,unigram in enumerate(unigram_set):
    unigram_idx[unigram] = idx
pickle.dump(unigram_idx,open('unigram_idx.pkl','wb'))

labfn = '../MultimodalFeatures/audioFeatures/cats.txt'
featlst = []
lablst = []
labdict = {'neutral':0,'positive':1,'negative':2}
for line in file(labfn):
    lst = line.split()
    feat = [0]*len(unigram_idx)
    raw_text = text_dict[lst[0]]
    for token in raw_text.split():
        feat[unigram_idx[token]]+=1
    featlst.append(feat)
    lablst.append(labdict[lst[1]])
X = np.array(featlst)
y = np.array(lablst)
print X.shape, y.shape
pickle.dump(X,open('text.pkl','wb'))
pickle.dump(y,open('lab.pkl','wb'))


