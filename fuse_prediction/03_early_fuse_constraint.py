### Ting-Yao Hu, 2016.04

import sys
import os
from sklearn.datasets import dump_svmlight_file
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
sys.path.append('/home2/tingyaoh/ML/PyAI/util')
from util_ml import *

def UncertaintyStats(X,y,pred_sec_lst,clf):
    type_str = str(type(clf))
    predlst = []
    scorelst = []
    ytestlst = []
    for Xtrain, ytrain, Xtest, ytest in KFold(X,y,5):
        for pred_idx, pred_sec in enumerate(pred_sec_lst):
            clf.fit(Xtrain[:,:,pred_idx],ytrain)
            ypred = clf.predict(Xtest[:,:,pred_idx])
            predlst+=ypred.tolist()
            if 'LogisticRegression' in type_str:
                scores = np.abs(clf.predict_proba(Xtest[:,:,pred_idx])[:,0]-0.5)
            elif 'SVC' in type_str:
                scores = clf.decision_function(Xtest[:,:,pred_idx])
            scorelst+=scores.tolist()
            ytestlst+=ytest.tolist()

    ypreds = np.array(predlst)
    yscores = np.array(scorelst)
    ytests = np.array(ytestlst)
    return ypreds, yscores ,ytests


X_ts = pickle.load(open('text_seq.pkl'))
X_as = pickle.load(open('audio_seq.pkl'))
X_vs = pickle.load(open('video_seq.pkl'))
y = pickle.load(open('lab.pkl'))
y_dummy = pickle.load(open('lab.pkl'))
l = pickle.load(open('length.pkl'))

pred_sec_lst = [1,2,3,4,5,6,7,8,9,10]
textdim = X_ts.shape[1]
audiodim = X_as.shape[1]
videodim = X_vs.shape[1]

np.random.seed(1234)
X_as, y_dummy = RandomPerm(X_as,y_dummy)
np.random.seed(1234)
X_vs, y_dummy = RandomPerm(X_vs,y_dummy)
np.random.seed(1234)
X_ts, y = RandomPerm(X_ts,y)

datanum = X_ts.shape[0]
ls = np.zeros((datanum,1,10))
for idx in range(10):
    ls[:,0,idx] = l

X_avs = np.concatenate((X_ts,X_as,X_vs,ls),axis=1)

thres = 7
ytest_total = []
ypred_total = []
lcount = 0
for Xtrain, ytrain, Xtest, ytest in KFold(X_avs,y,5):
    Xtr, Xts = Xtrain[:,:-1], Xtest[:,:-1]
    l_ts = Xtest[:,-1,0]
    """
    Xtr_all = Xtr[:,:,0]
    for idx in range(9): Xtr_all = np.concatenate((Xtr_all,Xtr[:,:,idx+1]),axis=0)
    ytr_all = np.tile(ytrain,(10))
    print Xtr_all.shape
    #print ytr_all.shape
    """

    clf = LogisticRegression(C=100000,penalty='l1')
    clf.fit(Xtr[:,:,thres-1],ytrain)
    ypred = clf.predict(Xts[:,:,thres-1])

    coef = np.abs(clf.coef_[0])
    featlst = range(len(coef))
    featlst = sorted(featlst, key=lambda k:coef[k], reverse=True)
    #print featlst
    
    for idx in range(len(ypred)):
        lcount+=min(thres,l_ts[idx])
    
    print accuracy_score(ytest,ypred)
    ytest_total+=ytest.tolist()
    ypred_total+=ypred.tolist()

print accuracy_score(ytest_total,ypred_total)
print lcount
