### Ting-Yao Hu, 2016.04

import sys
import glob
import string
sys.path.append('/home2/tingyaoh/ML/PyAI/util')
from util_ml import *
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def parse_sphinx4_align(s):
    lst = s.split('{')
    tokenlst = []
    endtimelst = []
    for l in lst[1:]:
        token = l.split(',')[0]
        if token[0]=='<': continue
        endtime = max(float(l.split(',')[2].split(':')[0].split('[')[-1]),0)/1000
        tokenlst.append(token)
        endtimelst.append(endtime)
    return tokenlst,endtimelst

def prediction_feature(infn):
    tokenlst = []
    endtimelst = []
    for line in file(infn):
        lst = line.split('{')
        if len(lst)==1: continue
        tlst,elst = parse_sphinx4_align(line)
        tokenlst+=tlst
        endtimelst+=elst
    return tokenlst,endtimelst
            

thres = 10.0

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

### unigram dictionary
unigramlst = []
for key, item in text_dict.iteritems():
    lst = item.split()
    unigramlst+=lst
unigram_set = list(set(unigramlst))
unigram_idx = {}
for idx,unigram in enumerate(unigram_set):
    unigram_idx[unigram] = idx

### feature extraction
aligndir = 'align_text/'
labfn = '../MultimodalFeatures/audioFeatures/cats.txt'
featlst = []
lablst = []
labdict = {'neutral':0,'positive':1,'negative':2}
for line in file(labfn):
    lst = line.split()
    feat = [0]*len(unigram_idx)
    tokenlst,endtimelst = prediction_feature(aligndir+lst[0]+'.txt')
    ### feature from false alignment
    for token,endtime in zip(tokenlst,endtimelst):
        if endtime<thres:
            #feat[unigram_idx[token]]+=1
            feat[unigram_idx[token]]=1

    ### feature from full observation
    if duration_dict[lst[0]]<thres:
        feat = [0]*len(unigram_idx)
        raw_text = text_dict[lst[0]]
        for token in raw_text.split():
            #feat[unigram_idx[token]]+=1
            feat[unigram_idx[token]]+=1
    
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
    #clf = LinearSVC(C=0.05,penalty='l1',dual=False)
    #clf = LinearSVC(C=0.04) #current best
    clf = LinearSVC(C=0.04) 
    #clf = LogisticRegression(C=10,penalty='l1')
    clf.fit(Xtrain,ytrain)
    ypred = clf.predict(Xtest)
    print accuracy_score(ypred,ytest)
    ytest_total+=ytest.tolist()
    ypred_total+=ypred.tolist()

print accuracy_score(ytest_total,ypred_total)
