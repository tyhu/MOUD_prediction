### Ting-Yao Hu, 2016.04

import sys
import os
from sklearn.datasets import dump_svmlight_file
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
sys.path.append('/home2/tingyaoh/ML/PyAI/util')
sys.path.append('/home2/tingyaoh/ML/PyAI/reinforcement')
from util_ml import *
from q_learning import *

def score_scale(score):
    if abs(score)<0.2: return 1
    elif abs(score)<0.5: return 2
    else: return 3

def score_scale2(score):
    if abs(score)<0.5: return 1
    else: return 2


def statefunc(currenttime,score_t,score_a,score_v,currpred_t,currpred_a,currpred_v,endbool):
    if endbool:
        state = [0,0,0,0,1]
        #state = [0,0,0,0,0,0,0,1]
        #state= [0,0,0,0,0,1]
        return tuple(state)
    else:
        agree = 1 if currpred_t==currpred_a and currpred_a==currpred_v else 0
        #state = [currenttime,currpred_t,currpred_a,currpred_v,0]
        state = [currenttime,currpred_t,currpred_a,currpred_v,0]
        #state = [currenttime,currpred_t,currpred_a,currpred_v,score_scale(score_t),score_scale2(score_a),score_scale(score_v),0]
        #state = [currenttime, agree, score_scale(score_t), score_scale2(score_a), score_scale(score_v),0]
        return tuple(state)

def statefunc_text(currenttime,score_t,score_a,score_v,currpred_t,currpred_a,currpred_v,endbool):
    if endbool:
        state = [0,0,0,1]
    else:
        state = [currenttime,currpred_t,score_scale2(abs(score_t)),0]
    return tuple(state)

def distance_func(state1,state2):
    if state1[-1]!=state2[-1]: return sys.float_info.max
    dist = 0
    for i in range(len(state1)-1):
        dist+=abs(state1[i]-state2[i])
    return dist

X_t = pickle.load(open('text.pkl'))
X_ts = pickle.load(open('text_seq.pkl'))
y = pickle.load(open('lab.pkl'))
y_dummy = pickle.load(open('lab.pkl'))
l = pickle.load(open('length.pkl'))

X_ts = np.concatenate((X_ts,X_t[:,:,np.newaxis]),axis=2)
np.random.seed(1234)
X_ts, y = RandomPerm(X_ts,y)

datanum = X_ts.shape[0]
ls = np.zeros((datanum,1,11))
for idx in range(11):
    ls[:,0,idx] = l
X_ts = np.concatenate((X_ts,l))

pred_sec_lst = [1,2,3,4,5,6,7,8,9,10]
#pred_sec_lst = [1,2,3,4,5]
totaltime = 0
step_cost = -0.2
#ytest_total = []
#ypred_total = []
ytestlst = []
yfinallst = []
for Xtrain, ytrain, Xtest, ytest in KFold(X_ts,y,5):

    ### collecting mdp history
    historylst = []
    for Xtrtr, ytrtr, Xtrts, ytrts in KFold(Xtrain,ytrain,5):
        tpredlst = []
        tscorelst= []
        for pred_idx, pred_sec in enumerate(pred_sec_lst):
            Xtrtr_t, Xtrts_t = Xtrtr[:,:,pred_idx],Xtrts[:,:,pred_idx]
            ls_trtr = Xtrtr[:,-1,0]

            ### text mdp feature
            clf_t = LinearSVC(C=0.04)
            clf_t.fit(Xtrtr_t[:,:,pred_idx],ytrtr)
            #clf_t.fit(Xtrtr_t[:,:,-1],ytrtr)
            ytrpred = clf_t.predict(Xtrts_t[:,:,pred_idx])
            tpredlst.append(ytrpred.tolist())
            #scores = clf_t.decision_function(Xtrts_t[:,:,pred_idx])
            scores = clf_t.decision_function(Xtrts_t[:,:,-1])
            tscorelst.append(scores.tolist())
            
            ### audio mdp feature
            clf_a = LogisticRegression(C=0.001)
            clf_a.fit(Xtrtr_a[:,:,pred_idx],ytrtr)
            #ytrpred = clf_a.predict(Xtrts_a[:,:,pred_idx])
            ytrpred = clf_a.predict(Xtrts_a[:,:,-1])
            apredlst.append(ytrpred.tolist())
            scores = np.abs(clf_a.predict_proba(Xtrts_a[:,:,pred_idx])[:,0]-0.5)
            ascorelst.append(scores.tolist())

            ### video mdp feature
            clf_v = SVC(gamma=0.001,C=10)
            #clf_v.fit(Xtrtr_v[:,:,pred_idx],ytrtr)
            clf_v.fit(Xtrtr_v[:,:,-1],ytrtr)
            ytrpred = clf_v.predict(Xtrts_v[:,:,pred_idx])
            vpredlst.append(ytrpred.tolist())
            scores = clf_v.decision_function(Xtrts_v[:,:,pred_idx])
            vscorelst.append(scores.tolist())
            
        ytrpreds_t,ytrpreds_a,ytrpreds_v = np.array(tpredlst),np.array(apredlst),np.array(vpredlst)
        ytrscore_t,ytrscore_a,ytrscore_v = np.array(tscorelst),np.array(ascorelst),np.array(vscorelst)
        
        prednum = ytrpreds_t.shape[0]
        datanum = ytrpreds_t.shape[1]
        ls_trts = Xtrts[:,-1,0]
        for i in range(datanum):
            for pred_idx, pred_sec in enumerate(pred_sec_lst):
                currenttime = pred_sec
                currpred_t,currpred_a,currpred_v = ytrpreds_t[pred_idx,i],ytrpreds_a[pred_idx,i],ytrpreds_v[pred_idx,i]
                #currpred = 2 if currpred_t+currpred_a+currpred_v>=5 else 1
                currpred = currpred_t
                score_t,score_a,score_v = ytrscore_t[pred_idx,i],ytrscore_a[pred_idx,i],ytrscore_v[pred_idx,i]
                answer = ytrts[i]
                endbool = 1 if pred_idx==prednum-1 or ls_trts[i]<pred_sec else 0
                #if pred_idx==prednum-1 or Xtrts_a[i,-1,pred_idx]==Xtrts_a[i,-1,pred_idx+1]:
                #if pred_idx==prednum-1 or ls_trtr[i]<pred_idx:
                #    endbool = 1
                reward = 1 if answer==currpred else -1
                state = statefunc_text(currenttime,score_t,score_a,score_v,currpred_t,currpred_a,currpred_v,endbool)
                if state[-1]==1:
                    action = 'y'
                    tup = (state,action,reward,None)
                    historylst.append(tup)
                    continue
                nexttime = pred_sec_lst[pred_idx+1]
                currpred_tn,currpred_an,currpred_vn = ytrpreds_t[pred_idx+1,i],ytrpreds_a[pred_idx+1,i],ytrpreds_v[pred_idx+1,i]
                score_tn,score_an,score_vn = ytrscore_t[pred_idx+1,i],ytrscore_a[pred_idx+1,i],ytrscore_v[pred_idx+1,i]
                #nextend = 1 if pred_idx==prednum-2 or Xtrts_a[i,-1,pred_idx+1]==Xtrts_a[i,-1,pred_idx+2] else 0
                nextend = 1 if pred_idx==prednum-2 or ls_trts[i]<pred_sec+1 else 0
                state2 = statefunc_text(nexttime,score_tn,score_an,score_vn,currpred_tn,currpred_an,currpred_vn,nextend)
                action = 'y'
                tup = (state,action,reward,None)
                historylst.append(tup)
                action = 'n'
                tup = (state,action,-step_cost,state2)
                historylst.append(tup)
        ### end of collecting mdp history
    
    ### model training
    print 'model training'
    i = 0
    Xtr_t, Xts_t = Xtrain[:,i:i+textdim],Xtest[:,i:i+textdim]
    i+=textdim
    Xtr_a, Xts_a = Xtrain[:,i:i+audiodim,:],Xtest[:,i:i+audiodim,:]
    i+=audiodim
    Xtr_v, Xts_v = Xtrain[:,i:i+videodim,:],Xtest[:,i:i+videodim,:]
    l_ts = Xtest[:,-1,0]
    #print l_ts
    
    clf_t_lst,clf_a_lst,clf_v_lst = [],[],[]
    for pred_idx, pred_sec in enumerate(pred_sec_lst):
        clf_t = LinearSVC(C=0.04)
        #clf_t.fit(Xtr_t[:,:,pred_idx],ytrain)
        clf_t.fit(Xtr_t[:,:,-1],ytrain)
        clf_t_lst.append(clf_t)
        clf_a = LogisticRegression(C=0.001)
        #clf_a.fit(Xtr_a[:,:,pred_idx],ytrain)
        clf_a.fit(Xtr_a[:,:,-1],ytrain)
        clf_a_lst.append(clf_a)
        clf_v = SVC(gamma=0.001,C=10)
        #clf_v.fit(Xtr_v[:,:,pred_idx],ytrain)
        clf_v.fit(Xtr_v[:,:,-1],ytrain)
        clf_v_lst.append(clf_v)
    ### end of model training

    ### mdp training
    print 'mdp training'
    mdp = MyMDP(alpha = 0.5, gamma=0.9, iternum = 500)
    mdp.init_from_history(historylst)
    mdp.q_learn(historylst)
    ### end of mdp training

    ### testing
    print 'testing'
    tpredlst, apredlst, vpredlst = [],[],[]
    tscorelst, ascorelst,vscorelst = [],[],[]
    for pred_idx, pred_sec in enumerate(pred_sec_lst):
        tpredlst.append(clf_t_lst[pred_idx].predict(Xts_t[:,:,pred_idx]).tolist())
        tscorelst.append(clf_t_lst[pred_idx].decision_function(Xts_t[:,:,pred_idx]))
        apredlst.append(clf_a_lst[pred_idx].predict(Xts_a[:,:,pred_idx]).tolist())
        ascorelst.append(np.abs(clf_a_lst[pred_idx].predict_proba(Xts_a[:,:,pred_idx])[:,0]-0.5))
        vpredlst.append(clf_v_lst[pred_idx].predict(Xts_v[:,:,pred_idx]).tolist())
        vscorelst.append(clf_v_lst[pred_idx].decision_function(Xts_v[:,:,pred_idx]))
    yfinal = np.zeros_like(ytest)
    datanum = yfinal.shape[0]
    for i in range(datanum):
        for pred_idx, pred_sec in enumerate(pred_sec_lst):
            currenttime = pred_sec
            score_t,score_a,score_v = tscorelst[pred_idx][i],ascorelst[pred_idx][i],vscorelst[pred_idx][i]
            currpred_t,currpred_a,currpred_v = tpredlst[pred_idx][i],apredlst[pred_idx][i],vpredlst[pred_idx][i]
            endbool = 1 if pred_idx==prednum-1 or l_ts[i]<pred_sec else 0
            state = statefunc_text(currenttime,score_t,score_a,score_v,currpred_t,currpred_a,currpred_v,endbool)
            if mdp.policy(state,distance_func=distance_func)=='y' or endbool:
                #currpred = 2 if currpred_t+currpred_a+currpred_v>=5 else 1
                currpred = currpred_t
                yfinal[i] = currpred
                totaltime+=min(currenttime,l_ts[i])
                #totaltime+=min(8,l_ts[i])
                #print 'time: ',currenttime
                #print state
                break
    yfinallst+=yfinal.tolist()
    ytestlst+=ytest.tolist()
print 'totaltime: ',totaltime
print accuracy_score(yfinallst,ytestlst)

