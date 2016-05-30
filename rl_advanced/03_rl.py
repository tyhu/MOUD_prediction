### Ting-Yao Hu, 2016.0

import sys
import os
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from myconfig import *
sys.path.append(util_dir)
sys.path.append(early_predict_dir)
sys.path.append(rl_dir)
from util_ml import *
from rl_feature_extraction import *
from q_learning import *

def distance_func(state1,state2):
    if state1[-1]!=state2[-1]: return sys.float_info.max
    dist = 0
    for i in range(len(state1)-1):
        dist+=abs(state1[i]-state2[i])
    return dist

stepcost = float(sys.argv[1])
#stepcost = 0.1
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

maxl = 10
np.random.seed(1234)
ls, y_dummy = RandomPerm(ls,y_dummy)
np.random.seed(1234)
X_as, y_dummy = RandomPerm(X_as,y_dummy)
np.random.seed(1234)
X_vs, y_dummy = RandomPerm(X_vs,y_dummy)
np.random.seed(1234)
X_ts, y = RandomPerm(X_ts,y)

X_avs = np.concatenate((X_as,ls),axis=1)
ypred_total,ytest_total = [],[]
lcount = 0
for Xtrain, ytrain, ltrain, Xtest, ytest, ltest in KFold_withl(X_avs,y,l,5):
    #clf = LogisticRegression(C=0.01)
    clf = LinearSVC(C=0.01,penalty='l1',dual=False)
    hist = score_hist(Xtrain,ytrain,ltrain,clf)
    historylst = rl_feature(Xtrain,ytrain,ltrain,clf,hist,stepcost)
    
    mdp = MyMDP(alpha = 0.5, gamma=0.9, iternum = 500)
    mdp.init_from_history(historylst)
    mdp.q_learn(historylst)

    clflst = []
    for idx in range(maxl):
        #clf = LogisticRegression(C=0.01)
        clf = LinearSVC(C=0.01,penalty='l1',dual=False)
        clf.fit(Xtrain[:,:,idx],ytrain)
        clflst.append(clf)
    
    tsdatanum = Xtest.shape[0]
    ypred = []
    #classes = 
    for idx in range(tsdatanum):
        X_sample = Xtest[idx,:,:]
        test_state_lst = rl_feature_test(X_sample,clflst,ltest[idx],hist)
        #print test_state_lst

        for jdx, state in enumerate(test_state_lst):
            endbool = state[-1]
            if mdp.policy(state,distance_func=distance_func)=='y' or endbool or jdx==4:
                ypred.append(state[0])
                lcount+=jdx+1
                print jdx+1
                break
    print accuracy_score(ypred,ytest)
    ypred_total+=ypred
    ytest_total+=ytest.tolist()
print lcount
print accuracy_score(ytest_total,ypred_total)


