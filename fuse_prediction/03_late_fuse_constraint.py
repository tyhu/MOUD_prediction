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

thres = 8
ytest_total = []
ypred_total = []
lcount = 0
for Xtrain, ytrain, Xtest, ytest in KFold(X_avs,y,5):
    i = 0
    Xtr_t, Xts_t = Xtrain[:,i:i+textdim],Xtest[:,i:i+textdim]
    i+=textdim
    Xtr_a, Xts_a = Xtrain[:,i:i+audiodim,:],Xtest[:,i:i+audiodim,:]
    i+=audiodim
    Xtr_v, Xts_v = Xtrain[:,i:i+videodim,:],Xtest[:,i:i+videodim,:]
    l_ts = Xtest[:,-1,0]
    


        

    ### text feature
    clf_t = LinearSVC(C=0.04)
    clf_t.fit(Xtr_t[:,:,thres-1],ytrain)
    ### audio feature
    clf_a = LogisticRegression(C=0.001)
    clf_a.fit(Xtr_a[:,:,thres-1],ytrain)
    ### video feature
    clf_v = SVC(gamma=0.001,C=10)
    clf_v.fit(Xtr_v[:,:,thres-1],ytrain)

    ypr_a = clf_a.predict(Xts_a[:,:,thres-1])
    yscore_a = clf_a.decision_function(Xts_a[:,:,thres-1])
    ypr_v = clf_v.predict(Xts_v[:,:,thres-1])
    yscore_v = clf_v.decision_function(Xts_v[:,:,thres-1])
    ypr_t = clf_t.predict(Xts_t[:,:,thres-1])

    """
    ### feature for fusion
    ### audio RL feature
    clf_a = LogisticRegression(C=0.001)
    ypreds_a, yscores_a, ytests_a = UncertaintyStats(Xtr_a,ytrain,pred_sec_lst,clf_a)
    ### video RL feature
    clf_v = SVC(gamma=0.001,C=10)
    ypreds_v, yscores_v, ytests_v = UncertaintyStats(Xtr_v,ytrain,pred_sec_lst,clf_v)
    Xfuse = np.c_[ypreds_a,yscores_a,ypreds_v,yscores_v]
    yfuse = ytests_v
    clf_fuse = LinearSVC(C=1)
    clf_fuse.fit(Xfuse,yfuse)
    Xts_fuse = np.c_[ypr_a,yscore_a,ypr_v,yscore_v]
    print Xts_fuse.shape
    ypred = clf_fuse.predict(Xts_fuse)
    """
    
    ypred = np.zeros_like(ypr_t)
        
    for idx in range(len(ypred)):
        if ypr_a[idx]==ypr_v[idx]:
            ypred[idx]=ypr_a[idx]
            #lcount+=min(thres,l_ts[idx])
        else:
            #if np.abs(yscore_a[idx])>3: ypred[idx] = ypr_a[idx]
            #else: ypred[idx] =ypr_v[idx]
            ypred[idx] = ypr_t[idx]
            #lcount+=l_ts[idx]
        #lcount+=l_ts[idx]
        lcount+=min(thres,l_ts[idx])
    

    print 'audio: ',accuracy_score(ypr_a,ytest)
    print 'video: ',accuracy_score(ypr_v,ytest)
    ytest_total+=ytest.tolist()
    ypred_total+=ypred.tolist()

print accuracy_score(ytest_total,ypred_total)
print lcount
