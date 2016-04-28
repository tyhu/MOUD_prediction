import sys
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from util_ml import *
from sklearn.metrics import accuracy_score
from q_learning import *

def statefunc(currenttime,score,currpred,endbool):
    score = abs(score)
    if endbool:
        state = [0,0,0,1]
        return tuple(state)
    else:
        if score<0.5: scoreidx = 0
        elif score<1.0: scoreidx = 1
        else: scoreidx = 2
        state = [currenttime,scoreidx,currpred,0]
        return tuple(state)


pred_sec_lst = range(1,11)
xlst = []
for pred_sec in pred_sec_lst:
    featfn = 'video_feat_'+str(pred_sec)+'.txt'
    feat = np.genfromtxt(featfn,delimiter=',',skip_header=0).astype('float')
    y = np.array(feat[:,0].tolist())
    y = y[np.where(y!=0)]
    x = feat[:,1:]
    x = x[np.where(y!=0)]
    xlst.append(x.tolist())
X = np.array(xlst)
X = np.swapaxes(X,0,2)
X = np.swapaxes(X,0,1)
### X.shape: (datanum,featnum,prediction_step_num)

np.random.seed(1234)
X, y = RandomPerm(X,y)

pred_sec_lst = [1,2,3,4,5,6,7,8,9,10]
totaltime = 0
step_cost = -0.15
ytest_total = []
ypred_total = []
ytestlst = []
yfinallst = []
for Xtrain, ytrain, Xtest, ytest in KFold(X,y,5):
    ### collecting mdp history
    historylst = []
    for Xtrtr, ytrtr, Xtrts, ytrts in KFold(Xtrain,ytrain,5):
        predlst = []
        scorelst = []

        ### use all feature sequence to train
        Xallseq = Xtrtr[:,:,0]
        yallseq = ytrtr.tolist()
        for idx in range(1,len(pred_sec_lst)):
            Xallseq = np.concatenate((Xallseq,Xtrtr[:,:,idx]),axis=0)
            yallseq+=ytrtr.tolist()
        yallseq = np.array(yallseq)

        for pred_idx, pred_sec in enumerate(pred_sec_lst):
            clf = SVC(gamma=0.01,C=100)
            #clf = LogisticRegression(C=10)
            #clf.fit(Xtrtr[:,:,pred_idx],ytrtr)
            #clf.fit(Xtrtr[:,:,-1],ytrtr)
            clf.fit(Xallseq,yallseq)
            ytrpred = clf.predict(Xtrts[:,:,pred_idx])
            predlst.append(ytrpred.tolist())
            scores = clf.decision_function(Xtrts[:,:,pred_idx])
            scorelst.append(scores.tolist())
        ytrpreds = np.array(predlst)
        ytrscore = np.array(scorelst)
        #print ytrscore
        #print ytrpreds.shape, ytrscore.shape, ytrts.shape

        prednum = len(pred_sec_lst)
        datanum = ytrpreds.shape[1]
        for i in range(datanum):
            for pred_idx, pred_sec in enumerate(pred_sec_lst):
                currenttime = pred_sec
                score = ytrscore[pred_idx,i]
                currpred = ytrpreds[pred_idx,i]
                answer = ytrts[i]
                endbool = 0
                if pred_idx==prednum-1 or Xtrts[i,-1,pred_idx]==Xtrts[i,-1,pred_idx+1]:
                    endbool = 1
                if answer==currpred: reward = 1
                else: reward = -1
                state = statefunc(currenttime,score,currpred,endbool)
                if state[-1]==1:
                    action = 'y'
                    tup = (state,action,reward,None)
                    historylst.append(tup)
                    continue
                nexttime = pred_sec_lst[pred_idx+1]
                nextscore = ytrscore[pred_idx+1,i]
                nextpred = ytrpreds[pred_idx+1,i]
                nextend = 0
                if pred_idx==prednum-2 or Xtrts[i,-1,pred_idx+1]==Xtrts[i,-1,pred_idx+2]:
                    nextend = 1
                state2 = statefunc(nexttime,nextscore,nextpred,nextend)
                action = 'y'
                tup = (state,action,reward,None)
                historylst.append(tup)
                action = 'n'
                tup = (state,action,-step_cost,state2)
                historylst.append(tup)
    
    ### end of collecting mdp history
    ### models training
    Xallseq = Xtrain[:,:,0]
    yallseq = ytrain.tolist()
    for idx in range(1,len(pred_sec_lst)):
        Xallseq = np.concatenate((Xallseq,Xtrain[:,:,idx]),axis=0)
        yallseq+=ytrain.tolist()
    yallseq = np.array(yallseq)
    print 'model training'
    clflst = []
    for pred_idx, pred_sec in enumerate(pred_sec_lst):
        clf = SVC(gamma=0.01,C=100)
        #clf = LogisticRegression(C=10)
        #clf.fit(Xtrain[:,:,pred_idx],ytrain)
        #clf.fit(Xtrain[:,:,-1],ytrain)
        clf.fit(Xallseq,yallseq)
        clflst.append(clf)

    ### mdp training
    print 'mdp training'
    mdp = myMDP(alpha = 0.5, gamma=0.9, iternum = 500)
    mdp.Init_From_History(historylst)
    mdp.QLearn(historylst)
    mdp.PrintPolicy()

    ### testing
    print 'testing'
    ypredlst = []
    yscorelst = []
    for pred_idx, pred_sec in enumerate(pred_sec_lst):
        ypredlst.append(clflst[pred_idx].predict(Xtest[:,:,pred_idx]).tolist())
        yscorelst.append(clflst[pred_idx].decision_function(Xtest[:,:,pred_idx]))
    yfinal = np.zeros_like(ytest)
    datanum = yfinal.shape[0]
    for i in range(datanum):
        for pred_idx, pred_sec in enumerate(pred_sec_lst):
            currtime = pred_sec
            score = yscorelst[pred_idx][i]
            currpred = ypredlst[pred_idx][i]
            endbool = 0
            if pred_idx==prednum-1 or Xtest[i,-1,pred_idx]==Xtest[i,-1,pred_idx+1]:
                endbool = 1
            state = statefunc(currtime, score, currpred, endbool)
            #if pred_sec==2:
            #if pred_sec==10 or endbool:
            if mdp.Policy(state)=='y' or endbool==1:
                yfinal[i] = currpred
                totaltime+=currtime
                #print 'time: ',currtime
                #print currpred, ytest[i]
                break
    print accuracy_score(ytest.tolist(),yfinal.tolist())       
    yfinallst+=yfinal.tolist()
    ytestlst+=ytest.tolist()
print 'totaltime: ',totaltime
print accuracy_score(yfinallst,ytestlst)

