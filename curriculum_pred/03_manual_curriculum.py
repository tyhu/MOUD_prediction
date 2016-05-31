### Ting-Yao Hu, 2016.05
### 

import sys
import os
from sklearn.datasets import dump_svmlight_file
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from myconfig import *
sys.path.append(util_dir)
from util_ml import *

thres = int(sys.argv[1])
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

print X_as.shape
print X_vs.shape
print X_ts.shape
#X_avs = np.concatenate((X_ts,X_as,X_vs,ls),axis=1)
X_avs = np.concatenate((X_as,ls),axis=1)

#thres = 1
ytest_total, ypred_total = [],[]
for Xtrain, ytrain, Xtest, ytest in KFold(X_avs,y,5):
    
    Xtr_final = None
    ytr_all = []
    Xtr, Xts = Xtrain[:,:-1], Xtest[:,:-1]
    l_tr = Xtrain[:,-1,0]
    trdatanum = Xtrain.shape[0]
    
    for idx in range(trdatanum):
        #start = max(0,thres-2)
        #end = start+2 if l_tr[idx]>thres+1 else start+1
        #start = 0
        start = min(thres-1,int(np.ceil(l_tr[idx])))
        end = min(int(np.ceil(l_tr[idx]))+1,10)
        for jdx in range(start,end):
            Xtr_final = np.concatenate((Xtr_final,Xtr[idx:idx+1,:,jdx]),axis=0) if Xtr_final is not None else Xtr[idx:idx+1,:,jdx]
            ytr_all.append(ytrain[idx])
    ytr_all = np.array(ytr_all)
    
    #ytr_all = ytrain
    #Xtr_final = Xtr[:,:,thres]
    print Xtr_final.shape
    print ytr_all.shape
    #clf = LogisticRegression(C=10000,penalty='l1')
    #clf = LogisticRegression(C=0.001,penalty='l1')
    clf = LogisticRegression(C=0.001)
    #clf = LinearSVC(C=0.01,penalty='l1',dual=False)
    clf.fit(Xtr_final,ytr_all)
    ypred = clf.predict(Xts[:,:,thres-1])
    print accuracy_score(ytest,ypred)
    ytest_total+=ytest.tolist()
    ypred_total+=ypred.tolist()

print accuracy_score(ytest_total,ypred_total)


