### Ting-Yao Hu

import sys
import os
from sklearn.datasets import dump_svmlight_file
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from myconfig import *
sys.path.append(util_dir)
sys.path.append(early_predict_dir)
from util_ml import *
from latent import *
import copy

K = float(sys.argv[1])

X_ts = pickle.load(open(feat_dir+'text_seq.pkl'))
X_as = pickle.load(open(feat_dir+'audio_seq.pkl'))
X_vs = pickle.load(open(feat_dir+'video_seq.pkl'))
y = pickle.load(open(feat_dir+'lab.pkl'))
y_dummy = pickle.load(open(feat_dir+'lab.pkl'))
l = pickle.load(open(feat_dir+'length.pkl'))

datanum = X_ts.shape[0]
ls = np.zeros((datanum,1,10))
for idx in range(10):
    ls[:,0,idx] = l

np.random.seed(1234)
ls, y_dummy = RandomPerm(ls,y_dummy)
np.random.seed(1234)
X_as, y_dummy = RandomPerm(X_as,y_dummy)
np.random.seed(1234)
X_vs, y_dummy = RandomPerm(X_vs,y_dummy)
np.random.seed(1234)
X_ts, y = RandomPerm(X_ts,y)
#X_avs = np.concatenate((X_ts,X_as,X_vs,ls),axis=1)
X_avs = np.concatenate((X_as,ls),axis=1)
"""
count = 0
for idx in range(X_avs.shape[0]):
    #print min(np.ceil(ls[idx,0,0]),10)
    b = np.array_equal(X_avs[idx,:-1,-1],X_avs[idx,:-1,min(np.ceil(ls[idx,0,0]),10)-1])
    print b
    if not b:
        print X_avs[idx,:-1,-1].tolist()
        print X_avs[idx,:-1,min(np.ceil(ls[idx,0,0]),10)-1].tolist()
        print '=='
"""
default_clf = LogisticRegression(C=0.001)
lcount = 0
maxl = 5
ytest_total = []
ypred_total = []
for Xtrain, ytrain, Xtest, ytest in KFold(X_avs,y,5):

    Xtr, Xts = Xtrain[:,:-1], Xtest[:,:-1]
    l_tr,l_ts = Xtrain[:,-1,0], Xtest[:,-1,0]
    trdatanum = Xtrain.shape[0]
    
    #clf2 = LogisticRegression(C=1000,penalty='l1')
    clf2 = copy.copy(default_clf)
    clf2.fit(Xtr[:,:,-1],ytrain)
    clf = latent_predict(Xtr,ytrain,l_tr,K,default_clf)
    tsdatanum = Xtest.shape[0]
    ypred = []
    for idx in range(tsdatanum):
        l = int(min(np.ceil(l_ts[idx]),maxl))
        for jdx in range(l):
            yts = clf.predict(Xts[idx:idx+1,:,jdx])[0]
            if yts==0: continue
            else:
                ypred.append(yts)
                lcount+=jdx+1
                break
        if yts==0:
            yts = clf2.predict(Xts[idx:idx+1,:,-1])[0]
            ypred.append(yts)
            lcount+=l
            
#    clf = LogisticRegression(C=10000,penalty='l1')
#    clf.fit(Xtr[:,:,-1],ytrain)
#    print clf.classes_
    #ypred = clf.predict(Xts[:,:,-1])

    ytest_total+=ytest.tolist()
    ypred_total+=ypred
    print accuracy_score(ypred,ytest)
print 'K = '+str(K)
print accuracy_score(ytest_total,ypred_total)
print lcount
