### Ting-Yao Hu, 2016.04

import sys
import os
from sklearn.datasets import dump_svmlight_file
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
sys.path.append('/home2/tingyaoh/ML/PyAI/util')
from util_ml import *

mkltrain = '/home2/tingyaoh/ML/spg-gmkl/svm-train'
mklpredict = '/home2/tingyaoh/ML/spg-gmkl/svm-predict'

X_t = pickle.load(open('text.pkl'))
X_a = pickle.load(open('audio.pkl'))
X_v = pickle.load(open('video.pkl'))
y = pickle.load(open('lab.pkl'))

X = np.c_[X_t,X_a,X_v]
#X = X_a

textdim = X_t.shape[1]
audiodim = X_a.shape[1]
videodim = X_v.shape[1]

np.random.seed(1234)
X, y = RandomPerm(X,y)

ytest_total = []
ypred_total = []
text_count = 0
video_count = 0
for Xtrain, ytrain, Xtest, ytest in KFold(X,y,5):
    i = 0
    Xtr_t, Xts_t = Xtrain[:,i:i+textdim],Xtest[:,i:i+textdim]
    i+=textdim
    Xtr_a, Xts_a = Xtrain[:,i:i+audiodim],Xtest[:,i:i+audiodim]
    i+=audiodim
    Xtr_v, Xts_v = Xtrain[:,i:i+videodim],Xtest[:,i:i+videodim]

    ### text feature
    clf_t = LinearSVC(C=0.04)
    clf_t.fit(Xtr_t,ytrain)
    ### audio feature
    clf_a = LogisticRegression(C=0.001)
    clf_a.fit(Xtr_a,ytrain)
    ### video feature
    clf_v = SVC(gamma=0.001,C=10)
    clf_v.fit(Xtr_v,ytrain)

    ypr_t = clf_t.predict(Xts_t)
    ypr_a = clf_a.predict(Xts_a)
    ysc_a = np.abs(clf_a.predict_proba(Xts_a)[:,0]-0.5)
    print ysc_a
    ypr_v = clf_v.predict(Xts_v)
    ypred = np.zeros_like(ypr_t)
    y_fuse = ypr_t+ypr_a+ypr_v

       
    for idx in range(len(ypred)):
        if y_fuse[idx]>4: ypred[idx]=2
        else: ypred[idx]=1
    
    
    """
    for idx in range(len(ypred)):
        if ysc_a[idx]>0.25: ypred[idx]=ypr_a[idx]
        else:
            video_count+=1
            if ypr_a[idx]==ypr_v[idx]: ypred[idx]=ypr_v[idx]
            else:
                text_count+=1
                if y_fuse[idx]>4: ypred[idx]=2
                else: ypred[idx]=1
    """
    
    

    print 'text: ',accuracy_score(ypr_t,ytest)
    print 'audio: ',accuracy_score(ypr_a,ytest)
    print 'video: ',accuracy_score(ypr_v,ytest)
    print 'fuse: ',accuracy_score(ypred,ytest)
    ytest_total+=ytest.tolist()
    ypred_total+=ypred.tolist()

print 'text count: ',text_count
print 'video count: ',video_count
print accuracy_score(ytest_total,ypred_total)
