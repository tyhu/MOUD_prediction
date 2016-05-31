### text feature extraction
### Ting-Yao Hu,2016.05

import sys
import glob
import string
from collections import defaultdict
sys.path.append('/home2/tingyaoh/ML/PyAI/util')
from util_ml import *
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from nltk.util import ngrams
from nltk.probability import FreqDist

text_dict = {}

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
        raw_text = line.split(';')[2][1:-1]
        raw_text = "".join([ch for ch in raw_text if ch not in string.punctuation])
        raw_text = raw_text.lower()
        text_dict[key_utt] = raw_text

### unigram dictionary
unigramlst, bigramlst = [],[]
for key, item in text_dict.iteritems():
    lst = item.split()
    unigramlst+=lst
    bigramlst+=ngrams(lst,2)

unigram_set = list(set(unigramlst))
unigram_idx = defaultdict(int)
for idx,unigram in enumerate(unigram_set):
    unigram_idx[unigram] = idx

bigram_idx = defaultdict(int)
freqdist = FreqDist(bigramlst)
### remove  sparse bigram
bigram_set = freqdist.most_common(200)
for idx,bigram in enumerate(bigram_set):
    bigram_idx[bigram] = idx

labfn = '../MultimodalFeatures/audioFeatures/cats.txt'
featlst = []
lablst = []
labdict = {'neutral':0,'positive':1,'negative':2}
for line in file(labfn):
    lst = line.split()
    feat = [0]*(len(unigram_idx)+len(bigram_idx))
    raw_text = text_dict[lst[0]]
    for token in raw_text.split():
        feat[unigram_idx[token]]+=1
        #feat[unigram_idx[token]] = 1
    ### bigram feature
    for bigram in ngrams(raw_text.split(),2):
        if bigram in bigram_set:
            feat[len(unigram_idx)+bigram_idx[bigram]]+=1

    featlst.append(feat)
    lablst.append(labdict[lst[1]])
X = np.array(featlst)
y = np.array(lablst)
print X.shape, y.shape

np.random.seed(1234)
X, y = RandomPerm(X,y)
ytest_total = []
ypred_total = []

for Xtrain, ytrain, Xtest, ytest in KFold(X,y,5):
    #clf = LinearSVC(C=100,penalty='l1',dual=False)
    #clf = LinearSVC(C=0.04) #current best
    clf = LinearSVC(C=0.04)
    #clf = LogisticRegression(C=10,penalty='l1')
    clf.fit(Xtrain,ytrain)
    ypred = clf.predict(Xtest)
    print accuracy_score(ypred,ytest)
    ytest_total+=ytest.tolist()
    ypred_total+=ypred.tolist()

print accuracy_score(ytest_total,ypred_total)

